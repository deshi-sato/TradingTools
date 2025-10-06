#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
stream_microbatch.py (PUSH→一次バッファ→イベント駆動features)
- kabu WebSocket PUSH を受信し raw_push へ即時永続化
- PushBuffer でリアルタイムイベントを保持し features_worker が逐次処理
- OrderbookSaver により板スナップショット(orderbook_snapshot)を継続保存
- features_stream へ f1..f6/score をイベントドリブンに INSERT
"""

import argparse
import json
import logging
import queue
import socket
import sqlite3
import sys
import threading
import time
import ctypes
import atexit
import os
from collections import deque
from datetime import datetime
from math import exp, tanh
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException
except Exception:
    create_connection = None
    WebSocketTimeoutException = type("WebSocketTimeoutException", (), {})
    WebSocketConnectionClosedException = type("WebSocketConnectionClosedException", (), {})

logger = logging.getLogger(__name__)

_singleton_handle = None
_pidfile_path: Optional[Path] = None


def _cleanup_pid():
    global _singleton_handle, _pidfile_path
    try:
        if _singleton_handle:
            ctypes.windll.kernel32.CloseHandle(_singleton_handle)
    except Exception:
        pass
    try:
        if _pidfile_path and _pidfile_path.exists():
            _pidfile_path.unlink()
    except Exception:
        pass


def _normalize_sym(s: str) -> str:
    return s.split("@", 1)[0].strip()


def singleton_guard(tag: str):
    """単一タスクを保証するローカルミューテックス。"""
    global _singleton_handle, _pidfile_path
    name = f"Global\{tag}"
    _singleton_handle = ctypes.windll.kernel32.CreateMutexW(None, False, name)
    if ctypes.GetLastError() == 183:
        print(f"[ERROR] {tag} already running", file=sys.stderr)
        sys.exit(1)
    pid_dir = Path("runtime/pids")
    pid_dir.mkdir(parents=True, exist_ok=True)
    _pidfile_path = pid_dir / f"{tag}.pid"
    try:
        _pidfile_path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass
    atexit.register(_cleanup_pid)


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _to_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def load_json_utf8(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


DDL_OB = """
CREATE TABLE IF NOT EXISTS orderbook_snapshot(
  ticker TEXT,
  ts TEXT,
  bid1 REAL,
  ask1 REAL,
  over_sell_qty INT,
  under_buy_qty INT,
  sell_top3 INT,
  buy_top3 INT
);
"""

DDL_RAW_PUSH = """
CREATE TABLE IF NOT EXISTS raw_push(
  t_recv REAL NOT NULL,
  topic TEXT,
  symbol TEXT,
  payload TEXT NOT NULL
);
"""

DDL_FEATURES_STREAM = """
CREATE TABLE IF NOT EXISTS features_stream(
  symbol TEXT NOT NULL,
  t_exec REAL NOT NULL,
  ver TEXT NOT NULL,
  f1 REAL,
  f2 REAL,
  f3 REAL,
  f4 REAL,
  f5 REAL,
  f6 REAL,
  score REAL,
  spread_ticks INTEGER,
  bid1 REAL,
  ask1 REAL,
  bidqty1 REAL,
  askqty1 REAL,
  PRIMARY KEY(symbol, t_exec)
);
"""

FEATURE_COLUMNS = (
    "symbol",
    "t_exec",
    "ver",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "score",
    "spread_ticks",
    "bid1",
    "ask1",
    "bidqty1",
    "askqty1",
)


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
    except sqlite3.DatabaseError:
        return []
    rows = cur.fetchall()
    return [row[1] for row in rows] if rows else []


def ensure_tables(db: str):
    conn = sqlite3.connect(db)
    try:
        with conn:
            conn.execute(DDL_OB)
            conn.execute("DROP TABLE IF EXISTS tick_batch")
            existing_cols = _get_table_columns(conn, "features_stream")
            if existing_cols and existing_cols != list(FEATURE_COLUMNS):
                conn.execute("DROP TABLE IF EXISTS features_stream")
            conn.execute(DDL_RAW_PUSH)
            conn.execute(DDL_FEATURES_STREAM)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_time ON features_stream(t_exec)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_push_ts ON raw_push(t_recv)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_push_symbol_ts ON raw_push(symbol, t_recv)")
    finally:
        conn.close()


def insert_orderbook_snapshot(conn: sqlite3.Connection, rows):
    if not rows:
        return
    conn.executemany(
        "INSERT INTO orderbook_snapshot (ticker, ts, bid1, ask1, over_sell_qty, under_buy_qty, sell_top3, buy_top3) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )


class PushBuffer:
    def __init__(self, maxsize: int):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._put = 0
        self._get = 0
        self._drop = 0
        self._highwater = 0

    def put(self, item: Any) -> None:
        while True:
            try:
                self._queue.put(item, timeout=0.001)
                with self._lock:
                    self._put += 1
                    size = self._queue.qsize()
                    if size > self._highwater:
                        self._highwater = size
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue
                with self._lock:
                    self._drop += 1

    def get(self, timeout: float = 0.1) -> Any:
        item = self._queue.get(timeout=timeout)
        with self._lock:
            self._get += 1
        return item

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def snapshot_stats(self, reset: bool = False) -> Dict[str, int]:
        with self._lock:
            stats = {
                "put": self._put,
                "get": self._get,
                "drop": self._drop,
                "qsize": self._queue.qsize(),
                "highwater": self._highwater,
            }
            if reset:
                self._put = 0
                self._get = 0
                self._drop = 0
                self._highwater = self._queue.qsize()
        return stats


def metrics_worker(buf: PushBuffer, period: float, stop_event: threading.Event):
    if period <= 0:
        return
    while not stop_event.wait(period):
        stats = buf.snapshot_stats(reset=True)
        logger.info(
            "pushbuf stats: put=%d get=%d drop=%d qsize=%d highwater=%d",
            stats["put"],
            stats["get"],
            stats["drop"],
            stats["qsize"],
            stats["highwater"],
        )
    stats = buf.snapshot_stats(reset=False)
    if stats["put"] or stats["get"] or stats["drop"]:
        logger.info(
            "pushbuf final stats: put=%d get=%d drop=%d qsize=%d highwater=%d",
            stats["put"],
            stats["get"],
            stats["drop"],
            stats["qsize"],
            stats["highwater"],
        )


class ScoreState:
    def __init__(self, tick: float = 1.0):
        self.sc = 0.0
        self.ts = 0.0
        self.tick = tick
        self.mp = deque(maxlen=4)
        self.b1: Optional[float] = None
        self.a1: Optional[float] = None


def calc_feats(st: ScoreState, evt: Dict[str, Any], now: float, tau: float = 20.0) -> Dict[str, Any]:
    bid = _to_float(evt.get("BidPrice")) or 0.0
    ask = _to_float(evt.get("AskPrice")) or 0.0
    b1 = _to_float(evt.get("BidQty1")) or 0.0
    a1 = _to_float(evt.get("AskQty1")) or 0.0

    spread_raw = evt.get("SpreadTicks")
    if spread_raw is None:
        tick = max(1e-6, st.tick)
        spread = int(round((ask - bid) / tick)) if tick else 0
    else:
        try:
            spread = int(float(spread_raw))
        except Exception:
            spread = 0
    spread = max(0, spread)

    dt = (now - st.ts) if st.ts else 0.0
    if dt > 0.0:
        st.sc *= exp(-dt / tau)

    f1 = max(0.0, 1.0 - min(3, spread) / 3.0)

    bsum = b1 + (_to_float(evt.get("BidQty2")) or 0.0) + (_to_float(evt.get("BidQty3")) or 0.0)
    asum = a1 + (_to_float(evt.get("AskQty2")) or 0.0) + (_to_float(evt.get("AskQty3")) or 0.0)
    denom = max(1.0, bsum + asum)
    f2 = (bsum - asum) / denom

    volume_side = b1 + a1
    mp = (ask * b1 + bid * a1) / max(1.0, volume_side)
    st.mp.append(mp)
    if len(st.mp) >= 3:
        f3 = tanh((st.mp[-1] - st.mp[-3]) / (2 * max(1e-6, st.tick)))
    else:
        f3 = 0.0

    iv = min(1.5, dt) if dt > 0.0 else 1.5
    f4 = 1.0 - iv / 1.5

    mo_b = _to_float(evt.get("MarketOrderBuyQty")) or 0.0
    mo_s = _to_float(evt.get("MarketOrderSellQty")) or 0.0
    mos = mo_b + mo_s
    f5 = (mo_b - mo_s) / mos if mos > 0 else 0.0

    if st.b1 is None or st.a1 is None:
        f6 = 0.0
    else:
        db = b1 - st.b1
        da = a1 - st.a1
        v = (-db + da)
        if v > 0:
            f6 = 1.0
        elif v < 0:
            f6 = -1.0
        else:
            f6 = 0.0

    delta = 2 * f1 + 2 * f2 + 2 * f3 + 1 * f4 + 2 * f5 + 1 * f6
    if spread >= 3:
        delta -= 1.0

    st.sc = max(0.0, min(10.0, st.sc + delta))
    st.ts = now
    st.b1 = b1
    st.a1 = a1

    symbol = str(evt.get("Symbol") or evt.get("sym") or "")
    return {
        "symbol": symbol,
        "t_exec": now,
        "ver": "feat_v1",
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f5": f5,
        "f6": f6,
        "score": st.sc,
        "spread_ticks": spread,
        "bid1": bid,
        "ask1": ask,
        "bidqty1": b1,
        "askqty1": a1,
    }


def insert_features_row(conn: sqlite3.Connection, d: Dict[str, Any]):
    conn.execute(
        "INSERT OR IGNORE INTO features_stream(symbol,t_exec,ver,f1,f2,f3,f4,f5,f6,score,spread_ticks,bid1,ask1,bidqty1,askqty1) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            d["symbol"],
            d["t_exec"],
            d["ver"],
            d["f1"],
            d["f2"],
            d["f3"],
            d["f4"],
            d["f5"],
            d["f6"],
            d["score"],
            d["spread_ticks"],
            d["bid1"],
            d["ask1"],
            d["bidqty1"],
            d["askqty1"],
        ),
    )


def features_worker(buf: PushBuffer, db_path: str, board_q: Optional[queue.Queue], stop_event: threading.Event):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    states: Dict[str, ScoreState] = {}
    batch: list[Dict[str, Any]] = []
    last_flush = time.monotonic()

    def flush(force: bool = False):
        nonlocal batch, last_flush
        if not batch:
            if force:
                last_flush = time.monotonic()
            return
        try:
            with conn:
                for row in batch:
                    insert_features_row(conn, row)
        except Exception as exc:
            logger.exception("features_stream insert failed: %s", exc)
        finally:
            batch.clear()
            last_flush = time.monotonic()

    try:
        while not stop_event.is_set() or not buf.empty():
            try:
                evt = buf.get(timeout=0.2)
            except queue.Empty:
                if time.monotonic() - last_flush >= 0.5:
                    flush(force=True)
                continue

            if not isinstance(evt, dict):
                continue

            sym = str(
                evt.get("Symbol")
                or evt.get("IssueCode")
                or evt.get("SymbolCode")
                or evt.get("SymbolName")
                or ""
            ).strip()
            sym = _normalize_sym(sym)
            if not sym:
                continue
            evt["Symbol"] = sym

            tick_size = evt.get("TickSize") or evt.get("Tick")
            state = states.setdefault(sym, ScoreState())
            if tick_size is not None:
                try:
                    state.tick = max(1e-6, float(tick_size))
                except Exception:
                    pass

            is_board = "BidPrice" in evt and "AskPrice" in evt
            if is_board and board_q is not None:
                snap = (
                    sym,
                    datetime.now().isoformat(timespec="milliseconds"),
                    _to_float(evt.get("BidPrice")),
                    _to_float(evt.get("AskPrice")),
                    _to_int(evt.get("OverSellQty")),
                    _to_int(evt.get("UnderBuyQty")),
                    sum(_to_int(evt.get(f"Sell{i}Qty")) for i in range(1, 4)),
                    sum(_to_int(evt.get(f"Buy{i}Qty")) for i in range(1, 4)),
                )
                try:
                    board_q.put_nowait(snap)
                except queue.Full:
                    pass

            if not is_board:
                continue

            now = time.time()
            try:
                feats = calc_feats(state, evt, now)
            except Exception as exc:
                logger.debug("calc_feats skipped for %s: %s", sym, exc, exc_info=True)
                continue

            if not feats.get("symbol"):
                continue

            batch.append(feats)
            if len(batch) >= 50 or time.monotonic() - last_flush >= 0.5:
                flush()
        flush(force=True)
    finally:
        conn.close()


class RawWS(threading.Thread):
    def __init__(
        self,
        url: str,
        headers,
        stop_event: threading.Event,
        buf: PushBuffer,
        db_path: str,
        recv_timeout: float = 15.0,
        keepalive_sec: float = 10.0,
    ):
        super().__init__(daemon=True)
        self.url = url
        self.headers = headers
        self.stop_event = stop_event
        self.buf = buf
        self.db_path = db_path
        self.recv_timeout = recv_timeout
        self.keepalive_sec = keepalive_sec

    def run(self):
        if create_connection is None:
            logger.error("websocket-client not available")
            return

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        raw_batch: list[tuple[Any, Optional[str], Optional[str], str]] = []
        last_flush = time.monotonic()

        def flush(force: bool = False):
            nonlocal raw_batch, last_flush
            if not raw_batch:
                if force:
                    last_flush = time.monotonic()
                return
            try:
                with conn:
                    conn.executemany(
                        "INSERT INTO raw_push(t_recv, topic, symbol, payload) VALUES(?,?,?,?)",
                        raw_batch,
                    )
            except Exception as exc:
                logger.exception("raw_push insert failed: %s", exc)
            finally:
                raw_batch.clear()
                last_flush = time.monotonic()

        backoff = 1.0
        ws = None
        try:
            while not self.stop_event.is_set():
                try:
                    ws = create_connection(
                        self.url,
                        header=self.headers,
                        timeout=6.0,
                        sockopt=(
                            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                            (socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024),
                        ),
                    )
                    ws.settimeout(self.recv_timeout)
                    logger.info("connected raw")
                    backoff = 1.0
                    last_ping = 0.0
                    missed = 0
                    while not self.stop_event.is_set():
                        now = time.time()
                        if now - last_ping >= self.keepalive_sec:
                            try:
                                ws.ping()
                                missed = 0
                            except Exception:
                                missed += 1
                            if missed >= 2:
                                raise RuntimeError("ping fail x2")
                            last_ping = now
                        try:
                            msg = ws.recv()
                        except WebSocketTimeoutException:
                            if time.monotonic() - last_flush >= 0.5:
                                flush(force=True)
                            continue
                        except (WebSocketConnectionClosedException, OSError):
                            raise RuntimeError("ws closed")
                        if not msg:
                            continue

                        text = msg.decode("utf-8", "replace") if isinstance(msg, bytes) else str(msg)
                        try:
                            obj = json.loads(text)
                        except Exception:
                            continue

                        t_recv = time.time()
                        topic_raw = obj.get("Topic") or obj.get("MessageType") or obj.get("TopicName")
                        topic = str(topic_raw) if topic_raw is not None else None
                        symbol_raw = (
                            obj.get("Symbol")
                            or obj.get("IssueCode")
                            or obj.get("SymbolCode")
                            or obj.get("SymbolName")
                        )
                        symbol = _normalize_sym(str(symbol_raw)) if symbol_raw else ""
                        raw_batch.append((t_recv, topic, symbol or None, text))
                        if len(raw_batch) >= 50 or time.monotonic() - last_flush >= 0.5:
                            flush()

                        if isinstance(obj, dict):
                            if symbol:
                                obj["Symbol"] = symbol
                            self.buf.put(obj)
                    flush(force=True)
                except Exception as exc:
                    if self.stop_event.is_set():
                        break
                    logger.warning("raw ws error: %s (reconnect %.1fs)", exc, backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                finally:
                    if ws:
                        try:
                            ws.close()
                        except Exception:
                            pass
                        ws = None
        finally:
            flush(force=True)
            conn.close()


class OrderbookSaver(threading.Thread):
    def __init__(self, db_path: str, board_q: queue.Queue, stop_event: threading.Event, batch_size: int = 100, flush_interval: float = 0.5):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.board_q = board_q
        self.stop_event = stop_event
        self.batch_size = batch_size
        self.flush_interval = flush_interval

    def run(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        batch = []
        last_flush = time.monotonic()
        try:
            while not self.stop_event.is_set() or not self.board_q.empty() or batch:
                try:
                    item = self.board_q.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty:
                    pass
                now = time.monotonic()
                if batch and (len(batch) >= self.batch_size or now - last_flush >= self.flush_interval or self.stop_event.is_set()):
                    insert_orderbook_snapshot(conn, batch)
                    conn.commit()
                    batch.clear()
                    last_flush = now
        except Exception as exc:
            logger.exception("orderbook saver error: %s", exc)
        finally:
            if batch:
                try:
                    insert_orderbook_snapshot(conn, batch)
                    conn.commit()
                except Exception as exc:
                    logger.exception("orderbook saver final flush failed: %s", exc)
            conn.close()


def main():
    singleton_guard("stream_microbatch")
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    args = ap.parse_args()

    cfg = load_json_utf8(args.Config)
    symbols = list(cfg.get("symbols", []))
    host = str(cfg.get("host", "localhost"))
    port = int(cfg.get("port", 18080))
    token = str(cfg.get("token") or "")
    if not symbols:
        sys.exit("ERROR: symbols empty")
    if not token:
        sys.exit("ERROR: token empty")

    db_path = cfg.get("db_path", "naut_market.db")
    log_path = cfg.get("log_path", "logs/stream_microbatch.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger.info("boot symbols=%s db=%s", symbols, db_path)

    ensure_tables(db_path)

    board_q = queue.Queue(maxsize=int(cfg.get("orderbook_queue_max", 10000)))
    buf = PushBuffer(maxsize=int(cfg.get("queue_max", 2000)))
    stop_event = threading.Event()

    metrics_period = float(cfg.get("metrics_log_sec", 60))
    metrics_thread = threading.Thread(target=metrics_worker, args=(buf, metrics_period, stop_event), daemon=True)
    metrics_thread.start()

    url = f"ws://{host}:{port}/kabusapi/websocket?filter=ALL"
    headers = [f"X-API-KEY: {token}"]
    rx = RawWS(url, headers, stop_event, buf, db_path)
    rx.start()

    feat_thread = threading.Thread(target=features_worker, args=(buf, db_path, board_q, stop_event), daemon=True)
    feat_thread.start()

    orderbook_saver = OrderbookSaver(db_path, board_q, stop_event)
    orderbook_saver.start()

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        stop_event.set()
        rx.join(1.0)
        feat_thread.join(1.0)
        orderbook_saver.join(1.0)
        metrics_thread.join(1.0)


if __name__ == "__main__":
    main()
