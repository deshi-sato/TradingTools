#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stream_microbatch.py (安定版, IssueCode 対応)
- kabu WebSocket PUSH からティックを受信し tick_batch / features_stream に書き込み
- orderbook_snapshot は ws_board_receiver 側が書く想定で、ここでは参照のみ
"""

import argparse
import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- helpers ----------
def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def spread_bp(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return (ask - bid) / ((ask + bid) / 2.0) * 10000.0

def depth_imbalance_calc(buy3: int, sell3: int) -> float:
    total = buy3 + sell3
    if total <= 0:
        return 0.0
    return (buy3 - sell3) / total

def uptick_ratio(upt: int, dwn: int) -> float:
    total = upt + dwn
    if total <= 0:
        return 0.0
    return upt / total

def load_json_utf8(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

# ---------- WebSocket ----------
try:
    from websocket import (
        create_connection,
        WebSocketTimeoutException,
        WebSocketConnectionClosedException,
    )
except Exception:
    create_connection = None
    WebSocketTimeoutException = type("WebSocketTimeoutException", (), {})
    WebSocketConnectionClosedException = type("WebSocketConnectionClosedException", (), {})

# ---------- logging ----------
logger = logging.getLogger(__name__)

# ---------- DB DDL ----------
DDL_TICK_BATCH = """
CREATE TABLE IF NOT EXISTS tick_batch (
  ticker TEXT,
  ts_window_start TEXT,
  ts_window_end   TEXT,
  ticks           INT,
  upticks         INT,
  downticks       INT,
  vol_sum         REAL,
  last_price      REAL
);
"""
DDL_FEAT = """
CREATE TABLE IF NOT EXISTS features_stream (
  ticker TEXT,
  ts     TEXT,
  uptick_ratio REAL,
  vol_sum      REAL,
  spread_bp    REAL,
  buy_top3     INT,
  sell_top3    INT,
  depth_imbalance REAL,
  burst_buy    INT,
  burst_sell   INT,
  burst_score  REAL,
  streak_len   INT,
  surge_vol_ratio REAL,
  last_signal_ts  TEXT
);
"""

def ensure_tables(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    with conn:
        conn.executescript(DDL_TICK_BATCH + DDL_FEAT)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_tb_sym_end ON tick_batch(ticker, ts_window_end)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_feat_sym_ts ON features_stream(ticker, ts)")
    conn.close()

def insert_tick_batch(conn, rows):
    if rows:
        conn.executemany("INSERT INTO tick_batch VALUES (?,?,?,?,?,?,?,?)", rows)

def insert_features(conn, rows):
    if not rows:
        return
    cols = ["ticker","ts","uptick_ratio","vol_sum","spread_bp","buy_top3","sell_top3",
            "depth_imbalance","burst_buy","burst_sell","burst_score","streak_len",
            "surge_vol_ratio","last_signal_ts"]
    sql = "INSERT INTO features_stream (" + ",".join(cols) + ") VALUES (" + ",".join(["?"]*len(cols)) + ")"
    data = [tuple(r.get(c) for c in cols) for r in rows]
    conn.executemany(sql, data)

# ---------- PUSH receiver ----------
class PushTickReceiver(threading.Thread):
    def __init__(self, symbols, q, stop_event, *, host, port, token, price_guard,
                 connect_timeout=6.0, recv_timeout=2.0):
        if create_connection is None:
            raise RuntimeError("websocket-client is required (pip install websocket-client)")
        super().__init__(daemon=True)
        self.symbols = list(symbols)
        self.symbols_set = set(self.symbols)
        self.q = q
        self.stop_event = stop_event
        self.url = f"ws://{host}:{port}/kabusapi/websocket"
        self.headers = [f"X-API-KEY: {token}"]
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self.price_guard = price_guard
        self.last_vol = {s: None for s in self.symbols}

    def run(self):
        backoff = 1.0
        while not self.stop_event.is_set():
            ws = None
            try:
                ws = create_connection(self.url, header=self.headers, timeout=self.connect_timeout)
                ws.settimeout(self.recv_timeout)
                logger.info("connected (symbols=%d)", len(self.symbols))
                backoff = 1.0
                while not self.stop_event.is_set():
                    try:
                        msg = ws.recv()
                    except WebSocketTimeoutException:
                        continue
                    except (WebSocketConnectionClosedException, OSError) as e:
                        raise RuntimeError(f"connection closed: {e}")
                    if not msg:
                        continue
                    self._on_msg(msg)
            except Exception as e:
                if self.stop_event.is_set():
                    break
                logger.warning("ws error: %s (reconnect in %.1fs)", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
            finally:
                try:
                    if ws:
                        ws.close()
                except Exception:
                    pass

    def _on_msg(self, payload: Any):
        try:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", "ignore")
            obj = json.loads(payload)
        except Exception:
            return

        # --- FIX: Symbol / IssueCode の両対応 ---
        sym = str(obj.get("Symbol") or obj.get("IssueCode") or "").strip()
        if not sym or (self.symbols_set and sym not in self.symbols_set):
            return

        price = _to_float(obj.get("CurrentPrice") or obj.get("Price"))
        if price is None:
            return

        # guard
        lo_hi = self.price_guard.get(sym, (None, None))
        if isinstance(lo_hi, (list, tuple)) and len(lo_hi) == 2:
            lo, hi = lo_hi
            if (lo is not None and price < lo) or (hi is not None and price > hi):
                return

        # 出来高差分
        tv_raw = obj.get("TradingVolume") or obj.get("Volume")
        qty = 0
        try:
            tv = int(float(tv_raw)) if tv_raw is not None else None
            if tv is not None:
                prev = self.last_vol.get(sym)
                self.last_vol[sym] = tv
                if prev is not None:
                    qty = max(0, tv - prev)
        except Exception:
            qty = 0

        ts = (
            obj.get("CurrentPriceTime")
            or obj.get("TradeTime")
            or obj.get("TransactTime")
            or obj.get("Time")
        )
        if ts is None:
            ts = datetime.now().isoformat(timespec="milliseconds")
        else:
            ts = str(ts)
            if "T" not in ts and " " in ts:
                ts = ts.replace(" ", "T", 1)
            if len(ts) == 8 and ts.count(":") == 2:
                ts = f"{datetime.now().strftime('%Y-%m-%d')}T{ts}"

        try:
            self.q.put((sym, float(price), int(qty), ts), timeout=0.02)
        except queue.Full:
            pass

# ---------- misc ----------
def within_market_window(spec: Optional[str]) -> bool:
    if not spec:
        return True
    try:
        s, e = spec.split("-")
        sh = datetime.strptime(s, "%H:%M").time()
        eh = datetime.strptime(e, "%H:%M").time()
        now = datetime.now().time()
        return sh <= now <= eh
    except Exception:
        return True

# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    args = ap.parse_args()

    cfg = load_json_utf8(args.Config)
    symbols = list(cfg.get("symbols", []))
    host = str(cfg.get("host","localhost"))
    port = int(cfg.get("port",18080))
    token = str(cfg.get("token") or "")
    if not symbols:
        print("ERROR: symbols empty", file=sys.stderr); sys.exit(2)
    if not token:
        print("ERROR: token empty", file=sys.stderr); sys.exit(2)

    price_guard   = cfg.get("price_guard", {})
    market_window = cfg.get("market_window")
    window_ms     = int(cfg.get("window_ms", 300))
    db_path       = cfg.get("db_path","rss_snapshot.db")
    log_path      = cfg.get("log_path","logs/stream_microbatch.log")

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )

    logger.info("boot symbols=%s db=%s", symbols, db_path)
    ensure_tables(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    q: "queue.Queue" = queue.Queue(maxsize=int(cfg.get("tick_queue_max", 20000)))
    stop_event = threading.Event()

    rx = PushTickReceiver(symbols, q, stop_event, host=host, port=port, token=token, price_guard=price_guard)
    rx.start()

    last_price: Dict[str, Optional[float]] = {s: None for s in symbols}
    window_s = window_ms / 1000.0
    next_cut = time.monotonic() + window_s

    try:
        while True:
            if market_window and not within_market_window(market_window):
                time.sleep(0.2)
                continue

            ticks_buf: Dict[str, List[Tuple[float,int,str]]] = {s: [] for s in symbols}

            while time.monotonic() < next_cut:
                try:
                    s,price,qty,ts_iso = q.get(timeout=0.01)
                    if s in ticks_buf:
                        ticks_buf[s].append((price, qty, ts_iso))
                except queue.Empty:
                    pass

            ts_start_iso = datetime.now().isoformat(timespec="milliseconds")
            tick_rows=[]; feat_rows=[]

            for s in symbols:
                arr = ticks_buf[s]
                if arr:
                    upt=dwn=0; vol=0.0
                    prev = last_price[s] if last_price[s] is not None else arr[0][0]
                    for price, qty, _ in arr:
                        if price>prev: upt+=1
                        elif price<prev: dwn+=1
                        prev = price; vol += qty
                    last_price[s]=arr[-1][0]
                    ts_end_iso = arr[-1][2]
                    tick_rows.append((s, ts_start_iso, ts_end_iso, len(arr), upt, dwn, vol, last_price[s]))

                    ob = conn.execute(
                        """select bid1, ask1, buy_top3, sell_top3
                           from orderbook_snapshot
                           where ticker=? and ts <= ?
                           order by ts desc limit 1""",
                        (s, ts_end_iso)
                    ).fetchone()
                    if ob:
                        b1,a1,bt,st = ob
                        spr = spread_bp(b1,a1)
                        imb = depth_imbalance_calc(int(bt), int(st))
                        feat_rows.append({
                            "ticker": s, "ts": ts_end_iso,
                            "uptick_ratio": uptick_ratio(upt,dwn),
                            "vol_sum": vol,
                            "spread_bp": spr,
                            "buy_top3": int(bt),
                            "sell_top3": int(st),
                            "depth_imbalance": imb,
                            "burst_buy": 0, "burst_sell": 0, "burst_score": 0.0,
                            "streak_len": 0, "surge_vol_ratio": 1.0, "last_signal_ts": ""
                        })

            with conn:
                insert_tick_batch(conn, tick_rows)
                insert_features(conn, feat_rows)

            if tick_rows:
                logger.info("batch ticks=%s feats=%s", sum(r[3] for r in tick_rows), len(feat_rows))

            now = time.monotonic(); next_cut += window_s
            if next_cut < now:
                next_cut = now + window_s

    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        stop_event.set()
        rx.join(timeout=1.0)
        conn.close()

if __name__ == "__main__":
    main()
