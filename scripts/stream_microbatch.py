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
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from math import exp
from pathlib import Path
from typing import Any, Dict, List, Optional

from dateutil import parser as dateutil_parser

from scripts.ensure_registry_schema import ensure_registry_schema

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback when zoneinfo unavailable
    ZoneInfo = None  # type: ignore[assignment]

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


VOLUME_SPIKE_SHORT_TAU = 2.0
VOLUME_SPIKE_LONG_TAU = 12.0
VOLUME_SPIKE_EPS = 1e-6
V_RATE_ALPHA_SHORT = 0.4
V_RATE_ALPHA_LONG = 0.1
V_RATE_EPS = 1e-6


def _collect_top3_quantities(evt: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Return (buy_top3_qty, sell_top3_qty) from various board representations."""
    def _extract_from_lists(keys: List[str]) -> list[tuple[float, float]]:
        levels: list[tuple[float, float]] = []
        for key in keys:
            data = evt.get(key)
            if not isinstance(data, list):
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                price = _to_float(row.get("Price") or row.get("price"))
                qty = _to_float(row.get("Qty") or row.get("Quantity") or row.get("qty"))
                if price is None or qty is None:
                    continue
                levels.append((float(price), max(float(qty), 0.0)))
            if levels:
                break
        return levels

    def _extract_from_ranked(prefix: str) -> list[tuple[float, float]]:
        levels: list[tuple[float, float]] = []
        for idx in range(1, 11):
            node = evt.get(f"{prefix}{idx}")
            if not isinstance(node, dict):
                continue
            price = _to_float(node.get("Price") or node.get("price"))
            qty = _to_float(node.get("Qty") or node.get("Quantity") or node.get("qty"))
            if price is None or qty is None:
                continue
            levels.append((float(price), max(float(qty), 0.0)))
        return levels

    def _extract_direct(price_prefix: str, qty_prefix: str) -> list[tuple[float, float]]:
        levels: list[tuple[float, float]] = []
        for idx in range(1, 11):
            qty = _to_float(evt.get(f"{qty_prefix}{idx}"))
            price = _to_float(evt.get(f"{price_prefix}{idx}"))
            if price is None or qty is None:
                continue
            levels.append((float(price), max(float(qty), 0.0)))
        return levels

    bids = (
        _extract_from_lists(["Bids", "buys"])
        or _extract_from_ranked("Buy")
        or _extract_direct("BidPrice", "BidQty")
        or _extract_direct("BuyPrice", "BuyQty")
    )
    asks = (
        _extract_from_lists(["Asks", "sells"])
        or _extract_from_ranked("Sell")
        or _extract_direct("AskPrice", "AskQty")
        or _extract_direct("SellPrice", "SellQty")
    )

    buy_top3 = (
        sum(q for _, q in sorted(bids, key=lambda item: item[0], reverse=True)[:3]) if bids else None
    )
    sell_top3 = sum(q for _, q in sorted(asks, key=lambda item: item[0])[:3]) if asks else None
    return buy_top3, sell_top3


def _ema_update(prev: Optional[float], value: float, dt: float, tau: float) -> float:
    if tau <= 0.0:
        return value
    if prev is None:
        return value
    if dt <= 0.0:
        return prev
    alpha = 1.0 - exp(-dt / tau)
    return prev + alpha * (value - prev)


def _parse_volume_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dt = dateutil_parser.isoparse(text)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=JST)
        return dt.timestamp()
    return None


def _update_volume_rate(state: "ScoreState", evt: Dict[str, Any], fallback_ts: float) -> None:
    vol_raw = evt.get("TradingVolume")
    if vol_raw is None:
        return
    vol_now = _to_float(vol_raw)
    if vol_now is None:
        return
    ts_raw = evt.get("TradingVolumeTime")
    ts_now = _parse_volume_timestamp(ts_raw)
    if ts_now is None:
        ts_now = fallback_ts
    prev_ts = state.vol_prev_ts
    prev_vol = state.vol_prev_total
    if prev_ts is None or prev_vol is None:
        state.vol_prev_total = vol_now
        state.vol_prev_ts = ts_now
        if state.vol_rate_ema_short is None:
            state.vol_rate_ema_short = 0.0
        if state.vol_rate_ema_long is None:
            state.vol_rate_ema_long = 0.0
        state.v_rate = 0.0
        return
    dv = max(0.0, vol_now - prev_vol)
    dt = max(1e-3, ts_now - prev_ts)
    rate = dv / dt
    if state.vol_rate_ema_short is None:
        state.vol_rate_ema_short = rate
    else:
        state.vol_rate_ema_short = (1.0 - V_RATE_ALPHA_SHORT) * state.vol_rate_ema_short + V_RATE_ALPHA_SHORT * rate
    if state.vol_rate_ema_long is None:
        state.vol_rate_ema_long = rate
    else:
        state.vol_rate_ema_long = (1.0 - V_RATE_ALPHA_LONG) * state.vol_rate_ema_long + V_RATE_ALPHA_LONG * rate
    denom = max(state.vol_rate_ema_long, V_RATE_EPS)
    if denom > 0:
        state.v_rate = max(0.0, state.vol_rate_ema_short / denom)
    else:
        state.v_rate = 0.0
    state.vol_prev_total = vol_now
    state.vol_prev_ts = ts_now


def load_json_utf8(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


DDL_DATASET_REGISTRY = """
CREATE TABLE IF NOT EXISTS dataset_registry (
  dataset_id TEXT PRIMARY KEY,
  db_path TEXT NOT NULL,
  source_db_path TEXT NOT NULL,
  build_tool TEXT,
  code_version TEXT,
  config_json TEXT DEFAULT '{}',
  db_sha1 TEXT NOT NULL,
  source_db_sha1 TEXT NOT NULL,
  regime_tag TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


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
  dataset_id TEXT,
  symbol TEXT NOT NULL,
  t_exec REAL NOT NULL,
  ts_ms INTEGER NOT NULL,
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
  v_spike REAL DEFAULT 0,
  v_rate REAL DEFAULT 0,
  f1_delta REAL DEFAULT 0,
  PRIMARY KEY(dataset_id, symbol, t_exec)
);
"""

FEATURE_COLUMNS = (
    "dataset_id",
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
    "v_spike",
    "v_rate",
    "f1_delta",
)


JST = timezone(timedelta(hours=9))
if ZoneInfo is not None:
    try:
        JST = ZoneInfo("Asia/Tokyo")
    except Exception:
        pass


def get_jst_now() -> datetime:
    return datetime.now(tz=JST)


def get_trading_date_str(now: Optional[datetime] = None) -> str:
    current = now or get_jst_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    else:
        current = current.astimezone(JST)
    return current.strftime("%Y%m%d")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_db_paths(cfg_db_path: Optional[str], trading_date: str) -> Dict[str, Path]:
    base_dir: Path
    if cfg_db_path:
        try:
            candidate = Path(cfg_db_path)
            if candidate.is_dir():
                base_dir = candidate
            else:
                base_dir = candidate.parent
        except Exception:
            base_dir = Path("db")
    else:
        base_dir = Path("db")
    refeed_name = f"naut_market_{trading_date}_refeed.db"
    refeed_path = (base_dir / refeed_name).resolve()
    return {"refeed": refeed_path}


def sha1_file(path: Path) -> str:
    try:
        if not path.exists():
            logger.warning("sha1 skipped; file missing: %s", path)
            return ""
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        logger.warning("sha1 failed for %s: %s", path, exc)
        return ""


def build_config_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    symbols = [str(s) for s in cfg.get("symbols", [])]
    snapshot["symbols_preview"] = symbols[:10]
    snapshot["symbols_total"] = len(symbols)
    for key in ("mode", "trading_start", "trading_end"):
        if key in cfg:
            snapshot[key] = cfg[key]
    for key in ("queue_max", "orderbook_queue_max", "NoSleep", "no_sleep"):
        if key in cfg:
            snapshot[key] = cfg[key]
    settings = cfg.get("settings")
    if isinstance(settings, dict):
        naut_settings = settings.get("naut")
        if isinstance(naut_settings, dict):
            snapshot["settings_naut"] = naut_settings
    watchlist = cfg.get("watchlist")
    if isinstance(watchlist, list):
        watchlist_entries = [str(s) for s in watchlist]
        snapshot["watchlist_preview"] = watchlist_entries[:10]
        snapshot["watchlist_total"] = len(watchlist_entries)
    burst_cfg = cfg.get("burst")
    if isinstance(burst_cfg, dict):
        snapshot["burst"] = burst_cfg
    return snapshot


def register_dataset(
    db_path: Path,
    config_dict: Dict[str, Any],
    code_version: str,
    now_jst: Optional[datetime] = None,
) -> tuple[Optional[str], bool]:
    timestamp = now_jst or get_jst_now()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=JST)
    else:
        timestamp = timestamp.astimezone(JST)
    dataset_id = f"REF{timestamp.strftime('%Y%m%d_%H%M')}"
    created_at = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    updated_at = created_at
    config_json = json.dumps(config_dict, ensure_ascii=False)
    regime_tag = str(config_dict.get("regime_tag", ""))

    db_abs_path = db_path.resolve()
    db_path_str = str(db_abs_path)
    db_sha1 = sha1_file(db_abs_path)

    source_db_path = config_dict.get("source_db_path")
    if source_db_path:
        source_db_path = str(Path(source_db_path).resolve())
    else:
        source_db_path = db_path_str
    source_db_sha1 = sha1_file(Path(source_db_path)) if source_db_path else db_sha1
    if not source_db_sha1:
        source_db_sha1 = db_sha1

    conn = sqlite3.connect(db_path_str)
    try:
        ensure_registry_schema(db_abs_path)
        with conn:
            conn.execute(
                """
                INSERT INTO dataset_registry (
                    dataset_id,
                    db_path,
                    source_db_path,
                    build_tool,
                    code_version,
                    config_json,
                    db_sha1,
                    source_db_sha1,
                    regime_tag,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    db_path=excluded.db_path,
                    source_db_path=excluded.source_db_path,
                    build_tool=excluded.build_tool,
                    code_version=excluded.code_version,
                    config_json=excluded.config_json,
                    db_sha1=excluded.db_sha1,
                    source_db_sha1=excluded.source_db_sha1,
                    regime_tag=excluded.regime_tag,
                    updated_at=excluded.updated_at
                """,
                (
                    dataset_id,
                    db_path_str,
                    source_db_path,
                    "stream_microbatch.py",
                    code_version or "unknown",
                    config_json,
                    db_sha1,
                    source_db_sha1,
                    regime_tag,
                    created_at,
                    updated_at,
                ),
            )
        return dataset_id, True
    except Exception as exc:
        logger.exception("dataset_registry registration failed: %s", exc)
        return None, False
    finally:
        conn.close()


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
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        with conn:
            conn.execute(DDL_OB)
            conn.execute("DROP TABLE IF EXISTS tick_batch")
            conn.execute(DDL_RAW_PUSH)
            conn.execute(DDL_FEATURES_STREAM)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_time ON features_stream(t_exec)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feat_ds_time ON features_stream(dataset_id, t_exec)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_push_ts ON raw_push(t_recv)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_push_symbol_ts ON raw_push(symbol, t_recv)")
        ensure_features_schema(conn)
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
        self.b1: Optional[float] = None
        self.a1: Optional[float] = None
        self.bid_prev: float = 0.0
        self.ask_prev: float = 0.0
        self.b1_prev: float = 1.0
        self.a1_prev: float = 1.0
        self.buy_top3_prev: float = 0.0
        self.sell_top3_prev: float = 0.0
        self.vol_ema_short: Optional[float] = None
        self.vol_ema_long: Optional[float] = None
        self.vol_prev_total: Optional[float] = None
        self.vol_prev_ts: Optional[float] = None
        self.vol_rate_ema_short: Optional[float] = None
        self.vol_rate_ema_long: Optional[float] = None
        self.v_rate: float = 0.0


def calc_feats(st: ScoreState, evt: Dict[str, Any], now: float, tau: float = 20.0) -> Dict[str, Any]:
    # kabu仕様: 英語ラベルが逆。ここで論理Ask/Bidにスワップ
    ask = _to_float(evt.get("BidPrice")) or 0.0
    bid = _to_float(evt.get("AskPrice")) or 0.0

    def _qty(keys: list[str], nested: str | None = None) -> float:
        for key in keys:
            if key in evt and evt[key] is not None:
                return _to_float(evt[key]) or 0.0
        if nested:
            node = evt.get(nested)
            if isinstance(node, dict):
                return _to_float(node.get("Qty")) or 0.0
        return 0.0

    # 論理解釈: Ask 側が BidQty 系、Bid 側が AskQty 系に入ってくる
    a1 = _qty(["BidQty1", "BidQty"], nested="Buy1")
    b1 = _qty(["AskQty1", "AskQty"], nested="Sell1")

    buy_top3_opt, sell_top3_opt = _collect_top3_quantities(evt)
    buy_incr = 0.0
    sell_incr = 0.0
    if buy_top3_opt is not None:
        buy_incr = max(buy_top3_opt - st.buy_top3_prev, 0.0)
    if sell_top3_opt is not None:
        sell_incr = max(sell_top3_opt - st.sell_top3_prev, 0.0)
    volume_pulse = max(0.0, buy_incr) + max(0.0, sell_incr)
    v_spike = 0.0

    spread_raw = evt.get("SpreadTicks")
    if spread_raw is None:
        tick = max(1e-6, st.tick)
        spread_ticks = int(round((ask - bid) / tick)) if tick else 0
    else:
        try:
            spread_ticks = int(float(spread_raw))
        except Exception:
            spread_ticks = 0
    spread_ticks = max(0, spread_ticks)

    dt = (now - st.ts) if st.ts else 0.0
    if dt > 0.0:
        st.sc *= exp(-dt / tau)

    if st.ts == 0.0 or dt > 60:
        st.sc = 5.0

    if bid <= 0.0:
        bid = st.bid_prev or 0.0
    if ask <= 0.0:
        ask = st.ask_prev or 0.0
    if b1 <= 0.0:
        b1 = st.b1_prev or 1.0
    if a1 <= 0.0:
        a1 = st.a1_prev or 1.0

    if dt < 0.1:
        dt = 0.1

    st.bid_prev, st.ask_prev, st.b1_prev, st.a1_prev = bid, ask, b1, a1
    if st.vol_ema_short is None:
        st.vol_ema_short = volume_pulse
    else:
        st.vol_ema_short = _ema_update(st.vol_ema_short, volume_pulse, dt, VOLUME_SPIKE_SHORT_TAU)
    if st.vol_ema_long is None:
        st.vol_ema_long = volume_pulse
    else:
        st.vol_ema_long = _ema_update(st.vol_ema_long, volume_pulse, dt, VOLUME_SPIKE_LONG_TAU)
    if st.vol_ema_long and st.vol_ema_long > VOLUME_SPIKE_EPS:
        v_spike = max(0.0, min(10.0, st.vol_ema_short / max(st.vol_ema_long, VOLUME_SPIKE_EPS)))
    else:
        v_spike = 0.0
    if buy_top3_opt is not None:
        st.buy_top3_prev = buy_top3_opt
    if sell_top3_opt is not None:
        st.sell_top3_prev = sell_top3_opt

    spread_abs = max(0.0, ask - bid)
    mid = (ask + bid) * 0.5 if (ask and bid) else 0.0
    # ---- 改良版スプレッドスコア ----
    spread_raw = (spread_abs / mid * 1000.0) if mid else 0.0
    v = max(0.0, min(10.0, 10.0 - spread_raw * 0.5))

    import random

    v += random.uniform(-0.2, 0.2)
    v = max(0.0, min(10.0, v))

    st.sc += (v - 5.0) * 0.2
    st.sc = max(0.0, min(10.0, st.sc))

    f1 = 10.0 if v > 9.5 else v
    f2 = 10.0 if spread_abs < 2.0 else max(0.0, 10.0 - spread_abs)
    f3 = st.sc
    f4 = max(0.0, min(10.0, 10.0 - abs(ask - bid)))
    f5 = f1 * 0.5 + f2 * 0.3 + f3 * 0.2
    f6 = (f1 + f2 + f3 + f4 + f5) / 5.0

    st.ts = now
    st.b1 = b1
    st.a1 = a1

    symbol = str(evt.get("Symbol") or evt.get("sym") or "")
    return {
        "symbol": symbol,
        "t_exec": now,
        "ts_ms": int(round(now * 1000)),
        "ver": "feat_v1",
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f5": f5,
        "f6": f6,
        "score": st.sc,
        "spread_ticks": spread_ticks,
        "bid1": bid,
        "ask1": ask,
        "bidqty1": b1,
        "askqty1": a1,
        "v_spike": v_spike,
        "v_rate": st.v_rate if st.v_rate is not None else 0.0,
    }


def insert_features_row(conn: sqlite3.Connection, d: Dict[str, Any], dataset_id: str):
    f1_delta = d.get("f1_delta", 0.0)
    ts_ms = d.get("ts_ms")
    if ts_ms is None and d.get("t_exec") is not None:
        try:
            ts_ms = int(round(float(d["t_exec"]) * 1000))
        except Exception:
            ts_ms = 0
    ts_ms = int(ts_ms or 0)
    conn.execute(
        "INSERT OR IGNORE INTO features_stream(dataset_id,symbol,t_exec,ts_ms,ver,f1,f2,f3,f4,f5,f6,score,spread_ticks,bid1,ask1,bidqty1,askqty1,v_spike,v_rate,f1_delta) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            dataset_id,
            d["symbol"],
            d["t_exec"],
            ts_ms,
            d.get("ver", "feat_v1"),
            d.get("f1"),
            d.get("f2"),
            d.get("f3"),
            d.get("f4"),
            d.get("f5"),
            d.get("f6"),
            d.get("score"),
            d.get("spread_ticks"),
            d.get("bid1"),
            d.get("ask1"),
            d.get("bidqty1"),
            d.get("askqty1"),
            d.get("v_spike"),
            d.get("v_rate"),
            f1_delta,
        ),
    )


def ensure_features_schema(conn: sqlite3.Connection) -> None:
    """Ensure backward compatible columns exist on features_stream."""
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(features_stream);")]
        added_ts_ms = False
        if "v_spike" not in columns:
            logger.info("migrating features_stream: adding v_spike column")
            conn.execute("ALTER TABLE features_stream ADD COLUMN v_spike REAL DEFAULT 0")
            conn.commit()
            columns.append("v_spike")
        if "v_rate" not in columns:
            logger.info("migrating features_stream: adding v_rate column")
            conn.execute("ALTER TABLE features_stream ADD COLUMN v_rate REAL DEFAULT 0")
            conn.commit()
            columns.append("v_rate")
        if "f1_delta" not in columns:
            logger.info("migrating features_stream: adding f1_delta column")
            conn.execute("ALTER TABLE features_stream ADD COLUMN f1_delta REAL DEFAULT 0")
            conn.commit()
            logger.info("migrating features_stream: f1_delta added")
        if "ts_ms" not in columns:
            logger.info("migrating features_stream: adding ts_ms column")
            conn.execute("ALTER TABLE features_stream ADD COLUMN ts_ms INTEGER")
            conn.commit()
            conn.execute(
                "UPDATE features_stream SET ts_ms = CAST(t_exec * 1000 AS INTEGER) "
                "WHERE ts_ms IS NULL OR ts_ms = 0"
            )
            conn.commit()
            columns.append("ts_ms")
            added_ts_ms = True
        if "ts_ms" in columns:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feat_symbol_ts_ms ON features_stream(symbol, ts_ms)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feat_ds_symbol_ts_ms ON features_stream(dataset_id, symbol, ts_ms)"
            )
            conn.commit()
            if added_ts_ms:
                logger.info("migrating features_stream: ts_ms column and indexes ready")
    except sqlite3.DatabaseError as exc:
        logger.warning("failed to ensure features_stream schema: %s", exc)


def _transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich batched features with deltas, smoothing, and score scaling."""
    if df.empty:
        return df

    if "symbol" in df.columns:
        group_keys = df.groupby("symbol", sort=False)
    else:
        group_keys = None

    if "f1" in df.columns:
        try:
            if group_keys is not None:
                df["f1_delta"] = (
                    group_keys["f1"]
                    .transform(lambda s: s.astype(float).diff().fillna(0.0))
                    .clip(-5, 5)
                )
            else:
                df["f1_delta"] = df["f1"].astype(float).diff().fillna(0.0).clip(-5, 5)
        except Exception:
            df["f1_delta"] = 0.0

    for col in ("f2", "f3"):
        if col not in df.columns:
            continue
        try:
            if group_keys is not None:
                df[col] = group_keys[col].transform(
                    lambda s: s.astype(float).rolling(window=5, min_periods=1).mean()
                )
            else:
                df[col] = (
                    df[col].astype(float).rolling(window=5, min_periods=1).mean()
                )
        except Exception:
            # leave the original values untouched on failure
            pass

    if "score" in df.columns:
        try:
            scores = df["score"].astype(float).to_numpy()
            low = float(np.nanpercentile(scores, 1))
            high = float(np.nanpercentile(scores, 99))
            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError("score percentiles invalid")
            if abs(high - low) < 1e-6:
                # distribution is effectively flat; keep original values
                pass
            else:
                denom = high - low
                scaled = np.clip((scores - low) / denom * 10.0, 0.0, 10.0)
                df["score"] = scaled
        except Exception:
            pass

    return df


def features_worker(buf: PushBuffer, db_path: str, board_q: Optional[queue.Queue], stop_event: threading.Event, dataset_id: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA synchronous=NORMAL;")
    ensure_features_schema(conn)
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
            df = pd.DataFrame.from_records(batch)
            if df.empty:
                batch.clear()
                last_flush = time.monotonic()
                return
            df["dataset_id"] = dataset_id
            df = _transform_features(df)
            defaults = {
                "dataset_id": dataset_id,
                "symbol": "",
                "t_exec": np.nan,
                "ts_ms": 0,
                "ver": "feat_v1",
                "f1": 0.0,
                "f2": 0.0,
                "f3": 0.0,
                "f4": 0.0,
                "f5": 0.0,
                "f6": 0.0,
                "score": 0.0,
                "spread_ticks": 0,
                "bid1": 0.0,
                "ask1": 0.0,
                "bidqty1": 0.0,
                "askqty1": 0.0,
                "v_spike": 0.0,
                "v_rate": 0.0,
                "f1_delta": 0.0,
            }
            for col, default in defaults.items():
                if col not in df.columns:
                    df[col] = default
            required_cols = [
                "dataset_id",
                "symbol",
                "t_exec",
                "ts_ms",
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
                "v_spike",
                "v_rate",
                "f1_delta",
            ]
            missing = [col for col in required_cols if col not in df.columns]
            for col in missing:
                df[col] = defaults.get(col, 0.0)
            df = df[required_cols]
            rows_to_insert = [tuple(row) for row in df.itertuples(index=False, name=None)]
            conn.executemany(
                "INSERT OR IGNORE INTO features_stream(dataset_id,symbol,t_exec,ts_ms,ver,f1,f2,f3,f4,f5,f6,score,spread_ticks,bid1,ask1,bidqty1,askqty1,v_spike,v_rate,f1_delta) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows_to_insert,
            )
            conn.commit()
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

            now = time.time()
            _update_volume_rate(state, evt, now)

            is_board = "BidPrice" in evt and "AskPrice" in evt
            if is_board and board_q is not None:
                ask_price = _to_float(evt.get("BidPrice"))
                bid_price = _to_float(evt.get("AskPrice"))

                def _top3(prefix: str) -> int:
                    total = 0
                    for i in (1, 2, 3):
                        node = evt.get(f"{prefix}{i}")
                        if isinstance(node, dict):
                            total += _to_int(node.get("Qty"))
                        else:
                            total += _to_int(evt.get(f"{prefix}{i}Qty"))
                    return total

                sell_top3 = _top3("Sell")
                buy_top3 = _top3("Buy")
                snap = (
                    sym,
                    datetime.now().isoformat(timespec="milliseconds"),
                    bid_price,
                    ask_price,
                    _to_int(evt.get("OverSellQty")),
                    _to_int(evt.get("UnderBuyQty")),
                    sell_top3,
                    buy_top3,
                )
                try:
                    board_q.put_nowait(snap)
                except queue.Full:
                    pass

            if not is_board:
                continue

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
    ap.add_argument("-CodeVersion", default="unknown")
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

    jst_now = get_jst_now()
    trading_date = get_trading_date_str(jst_now)
    db_paths = resolve_db_paths(cfg.get("db_path"), trading_date)
    refeed_path = db_paths["refeed"]
    ensure_parent_dir(refeed_path)
    db_path = str(refeed_path)

    log_path = cfg.get("log_path", "logs/stream_microbatch.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger.info("boot symbols=%s db=%s", symbols, db_path)

    print(f"[stream_microbatch] Trading date (JST): {jst_now.strftime('%Y-%m-%d')}")
    print(f"[stream_microbatch] Refeed DB: {db_path}")

    try:
        ensure_tables(db_path)
    except Exception as exc:
        logger.exception("failed to prepare database: %s", exc)
        print(f"[stream_microbatch] DB初期化失敗: {exc}")
        return

    dataset_id, inserted = register_dataset(
        refeed_path,
        build_config_snapshot(cfg),
        args.CodeVersion,
        now_jst=jst_now,
    )
    if dataset_id:
        status = "registered" if inserted else "already exists"
        print(f"[stream_microbatch] dataset_id: {dataset_id} ({status})")
        logger.info("dataset_registry %s dataset_id=%s", status, dataset_id)
    else:
        print("[stream_microbatch] dataset_registry 登録失敗")
        logger.error("dataset_registry registration failed; aborting start")
        return

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

    feat_thread = threading.Thread(target=features_worker, args=(buf, db_path, board_q, stop_event, dataset_id), daemon=True)
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
