
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stream_microbatch_rest_fixed.py
- CurrentPrice (PUSH) -> tick_batch
- Board (REST /board/{symbol}@1) -> orderbook_snapshot
- Features -> features_stream  (spread/top3/imbalance are always attempted)

Robustness:
- If REST board is momentarily empty, fall back to the latest DB snapshot to compute features.
- Invalid board (ask1<bid1 or <=0) is discarded.
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
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---- small local helpers (self-contained) ----
def spread_bp(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return (ask - bid) / ((ask + bid) / 2) * 10000.0

def top3_sum(qtys: Optional[Sequence[Any]]) -> int:
    try:
        if not isinstance(qtys, (list, tuple)):
            return 0
        s = 0
        for v in list(qtys)[:3]:
            try:
                s += int(float(v))
            except Exception:
                pass
        return int(s)
    except Exception:
        return 0

def depth_imbalance_calc(buy3: int, sell3: int) -> float:
    total = buy3 + sell3
    if total <= 0: return 0.0
    return (buy3 - sell3) / total

def uptick_ratio(upt: int, dwn: int) -> float:
    total = upt + dwn
    if total <= 0: return 0.0
    return upt / total

def load_json_utf8(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---- HTTP (REST) ----
from urllib.request import Request, urlopen

# ---- WebSocket (price PUSH) ----
try:
    from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException
except Exception:  # pragma: no cover
    create_connection = None
    WebSocketTimeoutException = WebSocketConnectionClosedException = None

logger = logging.getLogger(__name__)

# ---- DB ----
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

def insert_orderbook(conn, rows):
    if rows:
        conn.executemany("INSERT OR IGNORE INTO orderbook_snapshot VALUES (?,?,?,?,?,?,?)", rows)

def insert_features(conn, feat_rows):
    if not feat_rows: return
    cols = ["ticker","ts","uptick_ratio","vol_sum","spread_bp","buy_top3","sell_top3",
            "depth_imbalance","burst_buy","burst_sell","burst_score","streak_len",
            "surge_vol_ratio","last_signal_ts"]
    sql = "INSERT INTO features_stream (" + ",".join(cols) + ") VALUES (" + ",".join(["?"]*len(cols)) + ")"
    rows = [tuple(fr.get(c) for c in cols) for fr in feat_rows]
    conn.executemany(sql, rows)

# ---- utils ----
def _to_float(x):
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

def parse_price_guard_config(data: Any) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    guard = {}
    if not isinstance(data, Mapping): return guard
    for k, spec in data.items():
        lo = _to_float(spec.get("min")) if isinstance(spec, Mapping) else None
        hi = _to_float(spec.get("max")) if isinstance(spec, Mapping) else None
        guard[str(k)] = (lo, hi)
    return guard

# ---- PUSH receiver ----
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
        # simple guard
        lo, hi = self.price_guard.get(sym, (None, None))
        if (lo is not None and price < lo) or (hi is not None and price > hi):
            return
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
            self.q.put((sym, float(price), int(qty), str(ts)), timeout=0.05)
        except queue.Full:
            pass

# ---- REST board fetcher ----
@dataclass
class _BoardCache:
    ts_last: float = 0.0
    snap: Dict[str, Any] = None  # type: ignore

class BoardFetcher:
    def __init__(self, *, host: str, port: int, token: str, rest_poll_ms: int = 500):
        self.base = f"http://{host}:{port}/kabusapi/board"
        self.headers = {"X-API-KEY": token}
        self.rest_poll_s = max(0.1, rest_poll_ms/1000.0)
        self.cache: Dict[str,_BoardCache] = {}

    def get_board(self, symbol: str) -> Dict[str, Any]:
        now = time.monotonic()
        ent = self.cache.get(symbol)
        if ent and now - ent.ts_last < self.rest_poll_s and ent.snap is not None:
            return ent.snap
        url = f"{self.base}/{symbol}@1"
        try:
            req = Request(url, headers=self.headers)
            with urlopen(req, timeout=1.8) as r:
                payload = r.read()
            j = json.loads(payload.decode("utf-8", "ignore"))
        except Exception:
            j = {}
        snap = self._normalize(j)
        if ent is None:
            ent = _BoardCache()
            self.cache[symbol] = ent
        ent.ts_last = now
        ent.snap = snap
        return snap

    def _normalize(self, j: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(j, Mapping) or not j:
        return {}
    # price keys (support multiple variants)
    bid1 = _to_float(
        j.get("BidPrice")
        or j.get("BestBidPrice")
        or j.get("BidsPrice")
        or j.get("Bid")
        or j.get("Bid1")
    )
    ask1 = _to_float(
        j.get("AskPrice")
        or j.get("BestAskPrice")
        or j.get("AsksPrice")
        or j.get("Ask")
        or j.get("Ask1")
    )
    # ladder quantities
    bids = (
        j.get("BidQty")
        or j.get("BuyQty")
        or j.get("BidQuantity")
        or j.get("BuyQuantity")
        or []
    )
    asks = (
        j.get("AskQty")
        or j.get("SellQty")
        or j.get("AskQuantity")
        or j.get("SellQuantity")
        or []
    )
    bids = list(bids) if isinstance(bids, (list, tuple)) else []
    asks = list(asks) if isinstance(asks, (list, tuple)) else []
    if bid1 is None or ask1 is None or bid1 <= 0 or ask1 <= 0 or ask1 < bid1:
        return {}
    return {"bid1": bid1, "ask1": ask1, "bids": bids, "asks": asks}

# ---- market window ----
def within_market_window(spec: Optional[str]) -> bool:
    if not spec:
        return True
    try:
        s, e = spec.split("-")
        hh, mm = s.split(":"); sh = dtime(int(hh), int(mm))
        hh, mm = e.split(":"); eh = dtime(int(hh), int(mm))
        now = datetime.now().time()
        return sh <= now <= eh
    except Exception:
        return True

# ---- main ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    ap.add_argument("--mode", choices=["online", "mock"], default="online")
    ap.add_argument("--probe-board", action="store_true")
    args = ap.parse_args()

    cfg = load_json_utf8(args.Config)
    symbols = list(cfg.get("symbols", []))
    host = str(cfg.get("host", "localhost")); port = int(cfg.get("port", 18080))
    token = str(cfg.get("token") or os.environ.get("KABU_TOKEN") or os.environ.get("KABU_API_KEY") or "")
    if not symbols:
        print("ERROR: symbols empty", file=sys.stderr); sys.exit(2)
    if not token:
        print("ERROR: token empty", file=sys.stderr); sys.exit(2)

    price_guard = parse_price_guard_config(cfg.get("price_guard", {}))
    rest_poll_ms = int(cfg.get("rest_poll_ms", 500))
    market_window = cfg.get("market_window")
    window_ms = int(cfg.get("window_ms", 300))
    db_path = cfg.get("db_path", "rss_snapshot.db")
    log_path = cfg.get("log_path", "logs/stream_microbatch.log")

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )

    print(f"[BOOT] mode={args.mode} symbols={symbols} db={db_path}")
    ensure_tables(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;"); conn.execute("PRAGMA synchronous=NORMAL;")

    if args.probe_board:
        try:
            req = Request(f"http://{host}:{port}/kabusapi/board/{symbols[0]}@1", headers={"X-API-KEY": token})
            with urlopen(req, timeout=2) as r:
                r.read(64)
            print("[PROBE] board OK")
        except Exception as e:
            print(f"[PROBE] board error: {e}", file=sys.stderr)

    q: "queue.Queue" = queue.Queue(maxsize=int(cfg.get("tick_queue_max", 20000)))
    stop_event = threading.Event()

    if args.mode == "online":
        rx = PushTickReceiver(symbols, q, stop_event, host=host, port=port, token=token,
                              price_guard=price_guard)
    else:
        class Mock(PushTickReceiver):
            def run(self):
                import random
                base = {s: 1000.0 for s in self.symbols}
                while not self.stop_event.is_set():
                    for s in self.symbols:
                        b = base[s]; d = random.choice([-0.1, 0, 0.1])
                        p = round(b + d, 1); qty = random.randint(1, 5) * 100
                        self.q.put((s, p, qty, datetime.now().isoformat(timespec="milliseconds")))
                        base[s] = p
                    time.sleep(0.05)
        rx = Mock(symbols, q, stop_event, host=host, port=port, token=token, price_guard=price_guard)

    rx.start()
    board = BoardFetcher(host=host, port=port, token=token, rest_poll_ms=rest_poll_ms)

    # helpers
    def latest_board_from_db(sym: str) -> Optional[Dict[str, Any]]:
        row = conn.execute("""select bid1, ask1, buy_top3, sell_top3
                              from orderbook_snapshot where ticker=?
                              order by ts desc limit 1""", (sym,)).fetchone()
        if not row:
            return None
        b1, a1, bt, st = row
        if b1 is None or a1 is None or b1 <= 0 or a1 <= 0 or a1 < b1:
            return None
        return {"bid1": b1, "ask1": a1, "bids": [bt, 0, 0], "asks": [st, 0, 0]}

    last_price: Dict[str, Optional[float]] = {s: None for s in symbols}
    window_s = window_ms / 1000.0
    next_cut = time.monotonic() + window_s

    try:
        while True:
            if market_window and not within_market_window(market_window):
                time.sleep(0.2); continue

            ticks_buf: Dict[str, List[Tuple[float,int,str]]] = {s: [] for s in symbols}

            while time.monotonic() < next_cut:
                try:
                    s, price, qty, ts_iso = q.get(timeout=0.01)
                    if s in ticks_buf:
                        ticks_buf[s].append((price, qty, ts_iso))
                except queue.Empty:
                    pass

            ts_start_iso = datetime.now().isoformat(timespec="milliseconds")
            tick_rows=[]; ob_rows=[]; feat_rows=[]

            for s in symbols:
                arr = ticks_buf[s]
                # Always try to poll board once per window (for this symbol)
                ob = board.get_board(s)
                if ob:
                    b1, a1 = ob.get("bid1"), ob.get("ask1")
                    spr   = spread_bp(b1, a1)
                    buy3  = top3_sum(ob.get("bids"))
                    sell3 = top3_sum(ob.get("asks"))
                    ob_rows.append((s, ts_start_iso if not arr else arr[-1][2], b1, a1, spr, buy3, sell3))
                else:
                    b1 = a1 = None; spr = None; buy3 = sell3 = 0

                if arr:
                    upt=dwn=0; vol=0.0; prev = last_price[s] if last_price[s] is not None else arr[0][0]
                    for price, qty, _ in arr:
                        if price>prev: upt+=1
                        elif price<prev: dwn+=1
                        prev = price; vol += qty
                    last_price[s]=arr[-1][0]
                    ts_end_iso = arr[-1][2]
                    tick_rows.append((s, ts_start_iso, ts_end_iso, len(arr), upt, dwn, vol, last_price[s]))

                    # compute features using board or fallback
                    if b1 is None or a1 is None:
                        ob_db = latest_board_from_db(s)
                        if ob_db:
                            b1 = ob_db["bid1"]; a1 = ob_db["ask1"]
                            spr = spread_bp(b1, a1)
                            buy3 = top3_sum(ob_db.get("bids"))
                            sell3 = top3_sum(ob_db.get("asks"))
                    imb = depth_imbalance_calc(int(buy3), int(sell3))
                    feat_rows.append({
                        "ticker": s, "ts": ts_end_iso,
                        "uptick_ratio": uptick_ratio(upt, dwn),
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
                insert_orderbook(conn, ob_rows)
                insert_features(conn, feat_rows)

            logging.info("batch ticks=%s ob_snaps=%s feats=%s",
                         sum(r[3] for r in tick_rows) if tick_rows else 0,
                         len(ob_rows), len(feat_rows))

            now = time.monotonic(); next_cut += window_s
            if next_cut < now: next_cut = now + window_s

    except KeyboardInterrupt:
        logging.info("stopped")
    finally:
        stop_event.set()
        rx.join(timeout=1.0)
        conn.close()

if __name__ == "__main__":
    main()
