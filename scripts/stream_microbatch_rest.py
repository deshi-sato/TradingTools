
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stream_microbatch (REST board polling edition)

- PUSH (WebSocket) for current price -> microbatch -> tick_batch
- REST polling for board (/kabusapi/board/{symbol}@1) -> orderbook_snapshot
- features_stream is built from tick + board (board absent => safe defaults)

Notes:
- Minimal, self-contained BoardFetcher (REST) included here.
- Safe guards: price guard, invalid board guard, per-symbol poll interval.
- Keeps original CLI flags for backward compatibility.
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

# project-local imports
from scripts.common_config import load_json_utf8
from scripts.feature_calc import top3_sum, spread_bp, depth_imbalance, uptick_ratio

# ------- HTTP (REST) -------
from urllib.request import Request, urlopen
import urllib.error

# ------- optional websocket (price PUSH) -------
try:
    import websocket  # type: ignore
    from websocket import (
        WebSocketConnectionClosedException,
        WebSocketTimeoutException,
        create_connection,
    )
except ImportError:  # pragma: no cover
    websocket = None  # type: ignore
    WebSocketConnectionClosedException = None  # type: ignore
    WebSocketTimeoutException = None  # type: ignore
    create_connection = None  # type: ignore


logger = logging.getLogger(__name__)


# =========================
# DB helpers
# =========================
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
        # indexes (idempotent)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_tb_sym_end ON tick_batch(ticker, ts_window_end)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_feat_sym_ts ON features_stream(ticker, ts)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_obs ON orderbook_snapshot(ticker, ts)")
    conn.close()


def insert_tick_batch(conn: sqlite3.Connection, rows: List[Tuple]) -> None:
    if rows:
        conn.executemany("INSERT INTO tick_batch VALUES (?,?,?,?,?,?,?,?)", rows)


def insert_orderbook(conn: sqlite3.Connection, rows: List[Tuple]) -> None:
    if rows:
        # tolerate duplicates silently (IGNORE) to avoid unique violation storms
        conn.executemany(
            "INSERT OR IGNORE INTO orderbook_snapshot VALUES (?,?,?,?,?,?,?)", rows
        )


def insert_features(conn: sqlite3.Connection, feat_rows: List[Mapping[str, Any]]) -> None:
    if not feat_rows:
        return
    cols = [
        "ticker","ts","uptick_ratio","vol_sum","spread_bp","buy_top3","sell_top3",
        "depth_imbalance","burst_buy","burst_sell","burst_score","streak_len",
        "surge_vol_ratio","last_signal_ts"
    ]
    sql = "INSERT INTO features_stream (" + ",".join(cols) + ") VALUES (" + ",".join(["?"]*len(cols)) + ")"
    rows = [tuple(fr.get(c) for c in cols) for fr in feat_rows]
    conn.executemany(sql, rows)


# =========================
# Utilities
# =========================
def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            return float(s)
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    v = _to_float(value)
    return int(v) if v is not None else None


def parse_price_guard_config(data: Any) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    guard: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if not isinstance(data, Mapping):
        return guard
    for raw_symbol, spec in data.items():
        symbol = str(raw_symbol).strip()
        if not symbol:
            continue
        lo: Optional[float] = None
        hi: Optional[float] = None
        if isinstance(spec, Mapping):
            lo = _to_float(spec.get("min") if "min" in spec else spec.get("low"))
            hi = _to_float(spec.get("max") if "max" in spec else spec.get("high"))
        elif isinstance(spec, (list, tuple)):
            if len(spec) >= 1: lo = _to_float(spec[0])
            if len(spec) >= 2: hi = _to_float(spec[1])
        else:
            val = _to_float(spec)
            if val is not None:
                hi = val
        guard[symbol] = (lo, hi)
    return guard


# =========================
# Price PUSH receiver (WebSocket)
# =========================
class PushTickReceiver(threading.Thread):
    """Subscribe kabuステーション PUSH and emit normalized ticks."""

    def __init__(
        self,
        symbols: List[str],
        q: "queue.Queue",
        stop_event: threading.Event,
        *,
        host: str,
        port: int,
        token: str,
        price_guard: Dict[str, Tuple[Optional[float], Optional[float]]],
        connect_timeout: float = 6.0,
        recv_timeout: float = 2.0,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        guard_log_interval: float = 60.0,
    ) -> None:
        if websocket is None or create_connection is None:
            raise RuntimeError("websocket-client is required. pip install websocket-client")
        super().__init__(daemon=True)
        self.symbols = symbols
        self.symbols_set = set(symbols)
        self.q = q
        self.stop_event = stop_event
        self.url = f"ws://{host}:{port}/kabusapi/websocket"
        self.headers = [f"X-API-KEY: {token}"]
        self.price_guard = price_guard
        self.connect_timeout = max(1.0, connect_timeout)
        self.recv_timeout = max(0.5, recv_timeout)
        self.backoff_initial = max(0.5, backoff_initial)
        self.backoff_max = max(self.backoff_initial, backoff_max)
        self.guard_log_interval = max(5.0, guard_log_interval)
        self._last_volume: Dict[str, Optional[int]] = {s: None for s in symbols}
        self._guard_last_log: Dict[str, float] = {}
        self._ws = None
        self.logger = logging.getLogger(__name__ + ".PushTickReceiver")

    def run(self) -> None:
        backoff = self.backoff_initial
        while not self.stop_event.is_set():
            try:
                self._connect_and_stream()
                backoff = self.backoff_initial
            except Exception as exc:  # pragma: no cover
                if self.stop_event.is_set():
                    break
                self.logger.warning("PUSH connection error: %s (retry in %.1fs)", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self.backoff_max)
        if self._ws is not None:
            try: self._ws.close()
            except Exception: pass

    def _connect_and_stream(self) -> None:
        self.logger.info("connecting to %s", self.url)
        ws = create_connection(self.url, header=self.headers, timeout=self.connect_timeout)
        ws.settimeout(self.recv_timeout)
        self._ws = ws
        self.logger.info("connected (symbols=%s)", len(self.symbols))
        try:
            while not self.stop_event.is_set():
                try:
                    msg = ws.recv()
                except WebSocketTimeoutException:
                    continue
                except WebSocketConnectionClosedException as exc:
                    raise RuntimeError(f"connection closed: {exc}")
                except OSError as exc:
                    raise RuntimeError(f"socket error: {exc}")
                if not msg:
                    continue
                self._handle_payload(msg)
        finally:
            try: ws.close()
            except Exception: pass
            self.logger.info("disconnected")

    def _handle_payload(self, payload: Any) -> None:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", "ignore")
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return

        symbol = str(obj.get("Symbol") or obj.get("IssueCode") or "").strip()
        if not symbol or (self.symbols_set and symbol not in self.symbols_set):
            return

        price = _to_float(obj.get("CurrentPrice") or obj.get("Price"))
        if price is None:
            return
        if not self._price_allowed(symbol, price):
            return

        ts_value = obj.get("CurrentPriceTime") or obj.get("TradeTime") or obj.get("TransactTime") or obj.get("Time")
        ts_iso = self._normalize_timestamp(ts_value)

        total_volume = obj.get("TradingVolume") or obj.get("Volume")
        size = self._compute_volume_delta(symbol, total_volume)

        event = (symbol, float(price), size, ts_iso)
        try:
            self.q.put(event, timeout=0.05)
        except queue.Full:
            # drop silently, log at low frequency
            pass

    def _normalize_timestamp(self, value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat(timespec="milliseconds")
        if value is None:
            return datetime.now().isoformat(timespec="milliseconds")
        ts = str(value).strip()
        if not ts:
            return datetime.now().isoformat(timespec="milliseconds")
        if "T" not in ts and " " in ts:
            ts = ts.replace(" ", "T", 1)
        if len(ts) == 8 and ts.count(":") == 2:
            today = datetime.now().strftime("%Y-%m-%d")
            ts = f"{today}T{ts}"
        return ts

    def _compute_volume_delta(self, symbol: str, total: Any) -> int:
        total_int = _to_int(total)
        if total_int is None:
            return 0
        prev = self._last_volume.get(symbol)
        self._last_volume[symbol] = total_int
        if prev is None:
            return 0
        diff = total_int - prev
        return diff if diff > 0 else 0

    def _price_allowed(self, symbol: str, price: float) -> bool:
        guard = self.price_guard.get(symbol)
        if not guard:
            return True
        lo, hi = guard
        if lo is not None and price < lo:
            return False
        if hi is not None and price > hi:
            return False
        return True


# =========================
# REST BoardFetcher (self-contained)
# =========================
@dataclass
class _BoardCache:
    ts_last_fetch: float = 0.0
    snapshot: Dict[str, Any] = None  # type: ignore


class BoardFetcher:
    """
    Minimal REST polling fetcher.
    - /kabusapi/board/{symbol}@1
    - per-symbol poll interval (rest_poll_ms)
    - invalid board guard -> return {}
    """

    def __init__(self, *, host: str, port: int, token: str, rest_poll_ms: int = 500) -> None:
        self.base = f"http://{host}:{port}/kabusapi/board"
        self.headers = {"X-API-KEY": token}
        self.rest_poll_s = max(0.1, rest_poll_ms / 1000.0)
        self.cache: Dict[str, _BoardCache] = {}
        self.log = logging.getLogger(__name__ + ".BoardFetcher")

    def get_board(self, symbol: str) -> Dict[str, Any]:
        now = time.monotonic()
        ent = self.cache.get(symbol)
        if ent and (now - ent.ts_last_fetch) < self.rest_poll_s and ent.snapshot is not None:
            return ent.snapshot

        url = f"{self.base}/{symbol}@1"
        try:
            req = Request(url, headers=self.headers)
            with urlopen(req, timeout=1.8) as r:
                payload = r.read()
            j = json.loads(payload.decode("utf-8", "ignore"))
        except urllib.error.HTTPError as e:
            self.log.warning("board HTTP %s for %s", e.code, symbol)
            j = {}
        except Exception as e:
            self.log.warning("board error for %s: %s", symbol, e)
            j = {}

        snap = self._normalize_board(j)
        if ent is None:
            ent = _BoardCache()
            self.cache[symbol] = ent
        ent.ts_last_fetch = now
        ent.snapshot = snap
        return snap

    def _normalize_board(self, j: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(j, Mapping) or not j:
            return {}
        bid1 = _to_float(j.get("BidPrice"))
        ask1 = _to_float(j.get("AskPrice"))
        bids = j.get("BidQty") or []
        asks = j.get("AskQty") or []
        bids = list(bids) if isinstance(bids, (list, tuple)) else []
        asks = list(asks) if isinstance(asks, (list, tuple)) else []

        # invalid board guard
        if bid1 is None or ask1 is None or bid1 <= 0 or ask1 <= 0 or ask1 < bid1:
            return {}

        return {
            "bid1": bid1,
            "ask1": ask1,
            "bids": bids,
            "asks": asks,
        }


# =========================
# Market window helper
# =========================
def within_market_window(spec: Optional[str]) -> bool:
    if not spec:
        return True
    try:
        start_s, end_s = spec.split("-")

        def to_t(s: str) -> dtime:
            hh, mm = s.split(":")
            return dtime(int(hh), int(mm))

        now = datetime.now().time()
        return to_t(start_s) <= now <= to_t(end_s)
    except Exception:
        return True


# =========================
# Main worker
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    ap.add_argument("--mode", choices=["online", "mock"], help="online=PUSH, mock=random")
    ap.add_argument("--symbols", help="CSV 7203,9984,8306", default="")
    ap.add_argument("--probe-board", action="store_true", help="起動時に /board 疎通チェック")
    args = ap.parse_args()

    cfg = load_json_utf8(args.Config)

    mode_raw = args.mode or cfg.get("mode") or "online"
    mode = mode_raw.lower() if isinstance(mode_raw, str) else "online"
    if mode not in {"online", "mock"}:
        print(f"[WARN] unknown mode '{mode_raw}' -> fallback to 'online'", file=sys.stderr)
        mode = "online"

    symbols_cli = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else []
    symbols_cfg = list(cfg.get("symbols", [])) if isinstance(cfg.get("symbols"), list) else []
    symbols_final: List[str] = symbols_cli or symbols_cfg
    if not symbols_final:
        print("ERROR: symbols is empty. Specify --symbols or config.symbols[]", file=sys.stderr)
        sys.exit(2)

    host: str = str(cfg.get("host", "localhost")).strip() or "localhost"
    port: int = int(cfg.get("port", 18080))
    token: str = str(cfg.get("token") or "").strip()
    if not token:
        token = (os.environ.get("KABU_TOKEN") or os.environ.get("KABU_API_KEY") or "").strip()

    price_guard_cfg = parse_price_guard_config(cfg.get("price_guard", {}))
    ws_cfg = cfg.get("websocket") if isinstance(cfg.get("websocket"), Mapping) else {}
    receiver_kwargs: Dict[str, Any] = {}
    if isinstance(ws_cfg, Mapping):
        def _assign_float(src_key: str, dst_key: str) -> None:
            value = ws_cfg.get(src_key)
            if value is None:
                return
            try:
                receiver_kwargs[dst_key] = float(value)
            except (TypeError, ValueError):
                print(f"[WARN] websocket.{src_key}={value} is not numeric; ignoring", file=sys.stderr)
        _assign_float("connect_timeout", "connect_timeout")
        _assign_float("recv_timeout", "recv_timeout")
        _assign_float("backoff_initial", "backoff_initial")
        _assign_float("backoff_max", "backoff_max")
        _assign_float("guard_log_interval", "guard_log_interval")

    db_path: str = cfg.get("db_path", "rss_snapshot.db")
    log_path: str = cfg.get("log_path", "logs/stream_microbatch.log")
    rest_poll_ms: int = int(cfg.get("rest_poll_ms", 500))
    market_window: Optional[str] = cfg.get("market_window")
    window_ms: int = int(cfg.get("window_ms", 300))

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("stream_microbatch (REST board polling edition) start mode=%s host=%s port=%s", mode, host, port)
    logging.info("[RULE] board via REST polling rest_poll_ms=%s window_ms=%s", rest_poll_ms, window_ms)
    if price_guard_cfg:
        logging.info("[GUARD] price guard configured for %s symbols", len(price_guard_cfg))

    print(f"[BOOT] mode={mode} symbols={symbols_final} db_path={db_path} host={host}:{port}")

    if args.probe_board:
        try:
            url = f"http://localhost:{port}/kabusapi/board/{symbols_final[0]}@1"
            req = Request(url, headers={"X-API-KEY": token})
            with urlopen(req, timeout=3) as r:
                _ = r.read(64)
            print(f"[PROBE] /board {symbols_final[0]}@1 OK")
        except urllib.error.HTTPError as e:
            print(f"[PROBE] /board {symbols_final[0]}@1 HTTP {e.code}", file=sys.stderr)
        except Exception as e:
            print(f"[PROBE] /board error: {e}", file=sys.stderr)

    ensure_tables(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    q: "queue.Queue" = queue.Queue(maxsize=int(cfg.get("tick_queue_max", 20000)))
    stop_event = threading.Event()

    if mode == "online":
        if not token:
            print("ERROR: token is empty. Set config.token or KABU_TOKEN/KABU_API_KEY.", file=sys.stderr)
            sys.exit(2)
        try:
            tick_thread = PushTickReceiver(
                symbols_final, q, stop_event,
                host=host, port=port, token=token,
                price_guard=price_guard_cfg, **receiver_kwargs
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr); sys.exit(2)
    else:
        # minimal mock to keep interface
        class Mock(threading.Thread):
            def __init__(self, symbols, q, stop_event, interval_ms=50):
                super().__init__(daemon=True); self.symbols=symbols; self.q=q; self.stop=stop_event; self.dt=interval_ms/1000.0; self._p={s:1000.0 for s in symbols}
            def run(self):
                import random
                while not self.stop.is_set():
                    for s in self.symbols:
                        b=self._p[s]; d=random.choice([-0.1,0.0,0.1]); p=round(b+d,1); qty=random.randint(1,5)*100
                        self.q.put((s,p,qty,datetime.now().isoformat(timespec="milliseconds")))
                        self._p[s]=p
                    time.sleep(self.dt)
        mock_interval = int(cfg.get("mock_interval_ms", 50))
        tick_thread = Mock(symbols_final, q, stop_event, interval_ms=mock_interval)

    tick_thread.start()

    board = BoardFetcher(host=host, port=port, token=token, rest_poll_ms=rest_poll_ms)

    last_price: Dict[str, Optional[float]] = {s: None for s in symbols_final}
    window_s = window_ms / 1000.0
    next_cut = time.monotonic() + window_s

    try:
        while True:
            if market_window and not within_market_window(market_window):
                time.sleep(0.2)
                if datetime.now().time() > dtime(10, 0):
                    logging.info("out of market window: flushing & exit")
                    break

            # per-window collection buffers
            ticks_buf: Dict[str, List[Tuple[float, int, str]]] = {s: [] for s in symbols_final}

            # collect
            while time.monotonic() < next_cut:
                try:
                    s, price, qty, ts_iso = q.get(timeout=0.01)
                    if s in ticks_buf:
                        ticks_buf[s].append((price, qty, ts_iso))
                except queue.Empty:
                    pass

            # finalize window
            ts_start_iso = datetime.now().isoformat(timespec="milliseconds")
            tick_rows: List[Tuple] = []
            ob_rows: List[Tuple] = []
            feat_rows: List[Dict[str, Any]] = []

            for s in symbols_final:
                arr = ticks_buf[s]
                if not arr:
                    # still refresh board with ts_start
                    ob = board.get_board(s)
                    if ob:
                        b1, a1 = ob.get("bid1"), ob.get("ask1")
                        spr = spread_bp(b1, a1)
                        buy3 = top3_sum(ob.get("bids"))
                        sell3 = top3_sum(ob.get("asks"))
                        ob_rows.append((s, ts_start_iso, b1, a1, spr, buy3, sell3))
                    continue

                # aggregate tick batch
                upt = dwn = 0
                vol_sum = 0.0
                prev = last_price[s] if last_price[s] is not None else arr[0][0]
                for price, qty, _ in arr:
                    if price > prev: upt += 1
                    elif price < prev: dwn += 1
                    prev = price
                    vol_sum += qty
                last = arr[-1][0]
                last_price[s] = last

                ts_end_iso = arr[-1][2]
                tick_rows.append((s, ts_start_iso, ts_end_iso, len(arr), upt, dwn, vol_sum, last))

                # board snapshot (REST)
                ob = board.get_board(s)
                if ob:
                    b1, a1 = ob.get("bid1"), ob.get("ask1")
                    spr = spread_bp(b1, a1)
                    bids = ob.get("bids") or []
                    asks = ob.get("asks") or []
                    buy3 = top3_sum(bids)
                    sell3 = top3_sum(asks)
                    ob_rows.append((s, ts_end_iso, b1, a1, spr, buy3, sell3))
                else:
                    # board absent -> safe defaults
                    b1 = a1 = None
                    spr = None
                    buy3 = sell3 = 0

                # features
                feature_row = {
                    "ticker": s,
                    "ts": ts_end_iso,
                    "uptick_ratio": uptick_ratio(upt, dwn),
                    "vol_sum": vol_sum,
                    "spread_bp": spr,
                    "buy_top3": buy3,
                    "sell_top3": sell3,
                    "depth_imbalance": depth_imbalance(buy3, sell3),
                    "burst_buy": 0,
                    "burst_sell": 0,
                    "burst_score": 0.0,
                    "streak_len": 0,
                    "surge_vol_ratio": 1.0,
                    "last_signal_ts": "",
                }
                feat_rows.append(feature_row)

            # DB write
            with conn:
                insert_tick_batch(conn, tick_rows)
                insert_orderbook(conn, ob_rows)
                insert_features(conn, feat_rows)

            # metrics
            total_ticks = sum(r[3] for r in tick_rows) if tick_rows else 0
            logging.info("batch ticks=%s ob_snaps=%s feats=%s", total_ticks, len(ob_rows), len(feat_rows))

            # next window
            now_mono = time.monotonic()
            next_cut += window_s
            if next_cut < now_mono:
                next_cut = now_mono + window_s

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    finally:
        stop_event.set()
        tick_thread.join(timeout=1.0)
        conn.close()
        logging.info("stream_microbatch stop")


if __name__ == "__main__":
    main()
