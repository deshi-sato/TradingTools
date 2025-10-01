# scripts/ws_board_receiver.py
# -*- coding: utf-8 -*-
"""
Kabu WebSocket (Board / Tick) 受信ワーカー
- Board → orderbook_snapshot へ UPSERT
- Tick  → ticks_raw へ INSERT OR IGNORE
設定は config/stream_settings.json に統合：
  - token       : X-API-KEY に使用
  - db_path     : SQLite のフルパス
  - symbols     : 監視銘柄（ログ表示のみ）
  - price_guard : { "<code>": {"min": <float|null>, "max": <float|null>} }
ws_url はコード内に固定（ws://localhost:18080/kabusapi/websocket）
"""

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from websocket import create_connection  # pip install websocket-client
except Exception:
    create_connection = None  # type: ignore

WS_URL = "ws://localhost:18080/kabusapi/websocket"


def iso_now_ms() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def calc_spread_bp(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 10000.0


def top3_sum(arr: Optional[Sequence[Any]]) -> int:
    if not isinstance(arr, (list, tuple)):
        return 0
    s = 0
    for v in arr[:3]:
        try:
            s += int(float(v or 0))
        except Exception:
            pass
    return int(s)


def looks_like_dummy(bid1: Optional[float], ask1: Optional[float], buy3: int, sell3: int) -> bool:
    return bid1 == 1000.0 and ask1 == 1000.5 and buy3 == 2800 and sell3 == 2500


DDL = """
CREATE TABLE IF NOT EXISTS orderbook_snapshot (
  ticker     TEXT NOT NULL,
  ts         TEXT NOT NULL,
  bid1       REAL,
  ask1       REAL,
  buy_top3   INTEGER,
  sell_top3  INTEGER,
  spread_bp  REAL,
  total_bid  INTEGER,
  total_ask  INTEGER,
  src        TEXT,
  seq        INTEGER,
  PRIMARY KEY (ticker, ts)
);
CREATE TABLE IF NOT EXISTS ticks_raw (
  ticker   TEXT NOT NULL,
  ts       TEXT NOT NULL,
  price    REAL NOT NULL,
  size     INTEGER NOT NULL,
  seq      INTEGER,
  PRIMARY KEY (ticker, ts, price, size)
);
"""


def upsert_board(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO orderbook_snapshot
          (ticker, ts, bid1, ask1, buy_top3, sell_top3, spread_bp, total_bid, total_ask, src, seq)
        VALUES (:ticker, :ts, :bid1, :ask1, :buy_top3, :sell_top3, :spread_bp, :total_bid, :total_ask, :src, :seq)
        ON CONFLICT(ticker, ts) DO UPDATE SET
          bid1=excluded.bid1,
          ask1=excluded.ask1,
          buy_top3=excluded.buy_top3,
          sell_top3=excluded.sell_top3,
          spread_bp=excluded.spread_bp,
          total_bid=excluded.total_bid,
          total_ask=excluded.total_ask,
          src=excluded.src,
          seq=excluded.seq
        """,
        row,  # ★ ここが抜けていた
    )


def insert_tick(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO ticks_raw (ticker, ts, price, size, seq)
        VALUES (:ticker, :ts, :price, :size, :seq)
        """,
        row,  # ★ ここも抜けていた
    )


def load_json_utf8(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_guard_map(cfg: Dict[str, Any]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    raw = cfg.get("price_guard") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                lo = v.get("min")
                hi = v.get("max")
                lo_f = float(lo) if lo is not None else None
                hi_f = float(hi) if hi is not None else None
                out[str(k)] = (lo_f, hi_f)
    return out


def guard_allows(
    ticker: str,
    bid1: Optional[float],
    ask1: Optional[float],
    guard: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> bool:
    rng = guard.get(ticker)
    if not rng:
        return True
    lo, hi = rng
    for p in (bid1, ask1):
        if p is None:
            continue
        if (lo is not None and p < lo) or (hi is not None and p > hi):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-Config", required=True)
    parser.add_argument("-Verbose", type=int, default=1)
    args = parser.parse_args()

    cfg = load_json_utf8(args.Config)

    token = cfg.get("token") or cfg.get("kabu", {}).get("api_token")
    if not token:
        print("[ERROR] 'token' が未設定です。", file=sys.stderr)
        sys.exit(2)

    db_path = cfg.get("db_path")
    if not db_path or not os.path.isabs(db_path):
        print("[ERROR] 'db_path' はフルパスで設定してください。", file=sys.stderr)
        sys.exit(2)

    symbols = [str(s) for s in (cfg.get("symbols") or [])]
    guard_map = build_guard_map(cfg)

    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("ws_board_receiver")

    if create_connection is None:
        log.error("websocket-client が未インストールです。py -m pip install websocket-client")
        sys.exit(2)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(DDL)
    conn.commit()

    log.info("start ws receiver url=%s db=%s symbols=%s", WS_URL, db_path, symbols or "-")

    backoff = 1.0
    try:
        while True:
            ws = None
            try:
                ws = create_connection(WS_URL, header=[f"X-API-KEY: {token}"], timeout=10)
                log.info("connected")
                backoff = 1.0

                while True:
                    raw = ws.recv()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
                    except Exception:
                        continue

                    # ---- Board ----
                    if (
                        isinstance(payload, dict)
                        and "Symbol" in payload
                        and "BidPrice" in payload
                        and "AskPrice" in payload
                    ):
                        ticker = str(payload["Symbol"])
                        bid1 = float(payload["BidPrice"]) if payload.get("BidPrice") is not None else None
                        ask1 = float(payload["AskPrice"]) if payload.get("AskPrice") is not None else None
                        buy3 = top3_sum(payload.get("BuyQty") if isinstance(payload.get("BuyQty"), list) else None)
                        sell3 = top3_sum(payload.get("SellQty") if isinstance(payload.get("SellQty"), list) else None)

                        if looks_like_dummy(bid1, ask1, buy3, sell3):
                            continue
                        if not guard_allows(ticker, bid1, ask1, guard_map):
                            log.warning("guard skip %s: bid=%s ask=%s", ticker, bid1, ask1)
                            continue

                        row = {
                            "ticker": ticker,
                            "ts": payload.get("ExecutionDateTime") or payload.get("CurrentPriceTime") or iso_now_ms(),
                            "bid1": bid1,
                            "ask1": ask1,
                            "buy_top3": buy3,
                            "sell_top3": sell3,
                            "spread_bp": calc_spread_bp(bid1, ask1),
                            "total_bid": int(payload.get("TotalBidQty") or 0),
                            "total_ask": int(payload.get("TotalAskQty") or 0),
                            "src": "ws",
                            "seq": int(payload.get("SeqNum") or 0),
                        }
                        with conn:
                            upsert_board(conn, row)
                        continue

                    # ---- Tick ----
                    if (
                        isinstance(payload, dict)
                        and "Symbol" in payload
                        and "Price" in payload
                        and "Volume" in payload
                        and "ExecutionDateTime" in payload
                    ):
                        row = {
                            "ticker": str(payload["Symbol"]),
                            "ts": str(payload["ExecutionDateTime"]),
                            "price": float(payload["Price"]),
                            "size": int(payload["Volume"]),
                            "seq": int(payload.get("SeqNum") or 0),
                        }
                        with conn:
                            insert_tick(conn, row)
                        continue

            except Exception as e:
                if ws:
                    try:
                        ws.close()
                    except Exception:
                        pass
                wait = min(60.0, backoff) + random.random() * 0.5
                log.warning("ws error: %s; reconnect in %.1fs", e, wait)
                time.sleep(wait)
                backoff = min(60.0, backoff * 2.0)
            finally:
                if ws:
                    try:
                        ws.close()
                    except Exception:
                        pass
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
