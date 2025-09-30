#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stream_microbatch.py (REST/Mock削除版)

- CurrentPrice (PUSH via kabu WS) -> tick_batch
- features_stream は DB内の最新 orderbook_snapshot を参照して生成（本スクリプトは orderbook_snapshot を書かない）

依存:
  pip install websocket-client
設定:
  -Config JSON (symbols, host, port, token, market_window, window_ms, db_path, price_guard, tick_queue_max)
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
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

def top3_sum(qtys: Optional[Sequence[Any]]) -> int:
    if not isinstance(qtys, (list, tuple)):
        return 0
    s = 0
    for v in list(qtys)[:3]:
        try:
            s += int(float(v))
        except Exception:
            pass
    return int(s)

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

# ---------- WebSocket (price PUSH) ----------
try:
    from websocket import create_connection, WebSocketTimeoutException
except Exception as e:  # pragma: no cover
    create_connection = None
    WebSocketTimeoutException = type("WebSocketTimeoutException", (), {})  # dummy

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
DDL_OB_SNAP = """
CREATE TABLE IF NOT EXISTS orderbook_snapshot (
  ticker TEXT,
  ts     TEXT,
  bid1   REAL,
  ask1   REAL,
  spread_bp REAL,
  buy_top3  INT,
  sell_top3 INT
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
        conn.executescript(DDL_TICK_BATCH + DDL_OB_SNAP + DDL_FEAT)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_tb_sym_end ON tick_batch(ticker, ts_window_end)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_obs ON orderbook_snapshot(ticker, ts)")
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
    def __init__(self, symbols, q, stop_event, *, host, port, token, price_guard, connect_timeout=6.0, recv_timeout=2.0):
        if create_connection is None:
            raise RuntimeError("websocket-client is required (pip install websocket-client)")
        super().__init__(daemon=True)
        self.symbols = symbols
        self.q = q
        self.stop_event = stop_event
        self.url = f"ws://{host}:{port}/kabusapi/websocket"
        self.headers = [f"X-API-KEY: {token}"]
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self.price_guard = price_guard
        self.last_vol = {s: None for s in symbols}

    def run(self):
        while not self.stop_event.is_set():
            try:
                ws = create_connection(self.url, header=self.headers, timeout=self.connect_timeout)
                ws.settimeout(self.recv_timeout)
                logger.info("connected (symbols=%d)", len(self.symbols))
                while not self.stop_event.is_set():
                    try:
                        msg = ws.recv()
                    except WebSocketTimeoutException:
                        continue
                    if not msg:
                        continue
                    self._on_msg(msg)
            except Exception as e:
                logger.warning("ws error: %s", e)
                time.sleep(1.0)

    def _on_msg(self, payload: Any):
        try:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", "ignore")
            obj = json.loads(payload)
        except Exception:
            return
        sym = str(obj.get("Symbol") or obj.get("IssueCode") or "").strip()
        if not sym:
            return
        price = _to_float(obj.get("CurrentPrice") or obj.get("Price"))
        if price is None:
            return

        # sanity (price guard)
        lo_hi = self.price_guard.get(sym, (None, None))
        if isinstance(lo_hi, (list, tuple)) and len(lo_hi) == 2:
            lo, hi = lo_hi
            if (lo is not None and price < lo) or (hi is not None and price > hi):
                return

        # 約定数量の推定（TradingVolume 差分）
        tv = obj.get("TradingVolume") or obj.get("Volume")
        qty = 0
        try:
            tv = int(float(tv)) if tv is not None else None
            if tv is not None:
                prev = self.last_vol.get(sym)
                self.last_vol[sym] = tv
                qty = max(0, tv - prev) if prev is not None else 0
        except Exception:
            qty = 0

        ts = obj.get("CurrentPriceTime") or obj.get("TradeTime") or obj.get("Time") or datetime.now().isoformat(timespec="milliseconds")
        if "T" not in str(ts) and " " in str(ts):
            ts = str(ts).replace(" ", "T", 1)

        try:
            self.q.put((sym, float(price), int(qty), str(ts)), timeout=0.02)
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
    token = str(cfg.get("token") or os.environ.get("KABU_TOKEN") or os.environ.get("KABU_API_KEY") or "")
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

    # 板は DBに既に蓄積されている前提で、features_stream 生成時に「直近の板」を参照
    def latest_board_from_db(sym: str, ts_limit_iso: Optional[str]) -> Optional[Dict[str, Any]]:
        if ts_limit_iso:
            row = conn.execute(
                """select bid1, ask1, buy_top3, sell_top3
                   from orderbook_snapshot
                   where ticker=? and ts <= ?
                   order by ts desc limit 1""",
                (sym, ts_limit_iso)
            ).fetchone()
        else:
            row = conn.execute(
                """select bid1, ask1, buy_top3, sell_top3
                   from orderbook_snapshot
                   where ticker=?
                   order by ts desc limit 1""",
                (sym,)
            ).fetchone()
        if not row:
            return None
        b1, a1, bt, st = row
        if b1 is None or a1 is None or b1<=0 or a1<=0 or a1<b1:
            return None
        return {"bid1": b1, "ask1": a1, "buy3": int(bt), "sell3": int(st), "spr": spread_bp(b1,a1)}

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
                    # ティック集約
                    upt=dwn=0; vol=0.0
                    prev = last_price[s] if last_price[s] is not None else arr[0][0]
                    for price, qty, _ in arr:
                        if price>prev: upt+=1
                        elif price<prev: dwn+=1
                        prev = price; vol += qty
                    last_price[s]=arr[-1][0]
                    ts_end_iso = arr[-1][2]
                    tick_rows.append((s, ts_start_iso, ts_end_iso, len(arr), upt, dwn, vol, last_price[s]))

                    # 直近の板（DB参照のみ）
                    ob = latest_board_from_db(s, ts_end_iso)
                    if ob:
                        b1,a1 = ob["bid1"], ob["ask1"]
                        spr   = ob["spr"]
                        buy3  = ob["buy3"]
                        sell3 = ob["sell3"]
                    else:
                        b1=a1=None; spr=None; buy3=sell3=0

                    imb = depth_imbalance_calc(int(buy3), int(sell3))
                    feat_rows.append({
                        "ticker": s, "ts": ts_end_iso,
                        "uptick_ratio": uptick_ratio(upt,dwn),
                        "vol_sum": vol,
                        "spread_bp": spr,
                        "buy_top3": int(buy3),
                        "sell_top3": int(sell3),
                        "depth_imbalance": float(imb),
                        "burst_buy": 0, "burst_sell": 0, "burst_score": 0.0,
                        "streak_len": 0, "surge_vol_ratio": 1.0, "last_signal_ts": ""
                    })

            with conn:
                insert_tick_batch(conn, tick_rows)
                insert_features(conn, feat_rows)

            # 最小限のオペレーションログのみ残す
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
