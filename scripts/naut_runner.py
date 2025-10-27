#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
naut_runner.py

ナイトランナー: グリッドサーチ済み閾値(best_thresholds_*_latest.json)と
ランナー設定(config/naut_runner_*.json)を読み込み、features_stream を監視して
IFDOCO 模擬発注を実現する。paper と live(dry-run) を同一ロジックで扱い、
口座レジャーと観測ログを整備する。

Naut Runner consumes features emitted by stream_microbatch and executes
policy-driven IFDOCO instructions using threshold JSONs produced by
grid_search_thresholds_{buy|sell}. Paper and live brokers share the
same ledger and logging pipeline for auditability.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, time as dtime, timezone, timedelta
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from scripts.common_config import load_json_utf8

REPO_ROOT = Path(__file__).resolve().parents[1]
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"
ACTION_ENTRY = "entry"
ACTION_EXIT = "exit"

logger = logging.getLogger(__name__)

def _fmt_market_ts(ts):
    """data_ts(Unix秒/ミリ秒/None)を 'HH:MM:SS' に。未同期なら '' を返す。"""
    if ts is None:
        return ""
    if ts > 1e12:  # ms → s
        ts = ts / 1000.0
    # 2000-01-01 未満は未同期扱い（1970 を抑止）
    if ts < 946684800:
        return ""
    return datetime.fromtimestamp(ts, tz=JST).strftime("%H:%M:%S")

def _log_with_ts(level, msg, data_ts=None):
    t = _fmt_market_ts(data_ts)
    if t:
        logger.log(level, f"{t} {msg}")
    else:
        logger.log(level, msg)

JST = timezone(timedelta(hours=9))
class DataTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ts = getattr(record, "data_ts", 0.0)
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = 0.0
        # 2000-01-01 未満は未同期扱い → 時刻を出さない
        if ts <= 0.0 or ts < 946684800:
            return ""
        return datetime.fromtimestamp(ts, tz=JST).strftime("%H:%M:%S")

    def format(self, record):
        # 通常のformatを使った後で余計な先頭スペースを削る
        s = super().format(record)
        return s.lstrip()  # ← これでasctimeが空の時に出る空白を消す


_LATEST_DATA_TS: float = 0.0


def _update_latest_data_ts(candidate: Optional[float]) -> None:
    global _LATEST_DATA_TS
    if candidate is None:
        return
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        return
    if value <= 0.0:
        return
    if value >= _LATEST_DATA_TS:
        _LATEST_DATA_TS = value


def _get_latest_data_ts() -> float:
    return _LATEST_DATA_TS

_singleton_handle: Optional[int] = None
_pidfile_path: Optional[Path] = None

# ファイル先頭付近にユーティリティを追加
def _load_json_optional(path_str: str) -> dict:
    if not path_str:
        return {}
    p = Path(resolve_path(path_str))
    if not p.exists():
        logger.warning("Policy file not found: %s", p)
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        logger.exception("Failed to load policy JSON: %s", p)
        return {}

from dataclasses import fields, replace

def _apply_policy_overrides(cfg: RunnerConfig, pol: Dict[str, Any]) -> RunnerConfig:
    if not pol:
        return cfg

    # dataclass定義から “上書き可” のフィールド名集合を作る
    overridable = {
        f.name for f in fields(RunnerConfig)
        if f.metadata.get("overridable", True)  # 明示False以外は可
    }

    updates, changed = {}, []
    for k, v in pol.items():
        if v is None or k not in overridable or not hasattr(cfg, k):
            continue
        old = getattr(cfg, k)
        if old != v:
            updates[k] = v
            changed.append(f"{k}:{old}->{v}")

    if updates:
        logger.info("Policy overrides: %s", ", ".join(changed))
        return replace(cfg, **updates)   # frozen対策：新インスタンス
    return cfg
    # policy.json overrides applied on top of config defaults
    lot_size = safe_int(pol.get("lot_size"))
    min_qty = safe_int(pol.get("min_qty"))
    min_lot = safe_int(min_qty if min_qty is not None else lot_size, cfg.min_lot)

    def coalesce_float(key: str, default: float) -> float:
        value = safe_float(pol.get(key))
        return default if value is None else float(value)

    def coalesce_int(key: str, default: int) -> int:
        value = safe_int(pol.get(key))
        return default if value is None else int(value)

    def coalesce_buyup_mode(default: str) -> str:
        value = pol.get("buyup_mode")
        if value is None:
            return default
        candidate = str(value).strip().upper()
        if candidate not in {"EXIT", "HOLD", "TRAIL"}:
            return default
        return candidate

    return RunnerConfig(
        features_db=cfg.features_db,
        ops_db=cfg.ops_db,
        symbols=cfg.symbols,
        symbols_original=cfg.symbols_original,
        poll_interval_sec=cfg.poll_interval_sec,
        initial_cash=coalesce_float("initial_cash", cfg.initial_cash),
        fee_rate_bps=cfg.fee_rate_bps,
        slippage_ticks=cfg.slippage_ticks,
        tick_size=cfg.tick_size,
        tick_value=cfg.tick_value,
        min_lot=min_lot if min_lot is not None else cfg.min_lot,
        risk_per_trade_pct=coalesce_float("risk_per_trade_pct", cfg.risk_per_trade_pct),
        max_cash_per_trade=coalesce_float("max_cash_per_trade", cfg.max_cash_per_trade),
        max_concurrent_positions=cfg.max_concurrent_positions,
        daily_loss_limit_pct=coalesce_float("daily_loss_limit_pct", cfg.daily_loss_limit_pct),
        stats_interval_sec=cfg.stats_interval_sec,
        stop_loss_ticks=cfg.stop_loss_ticks,
        log_path=cfg.log_path,
        timezone=cfg.timezone,
        killswitch_check_interval_sec=cfg.killswitch_check_interval_sec,
        market_window=cfg.market_window,
        stop_loss_pct=coalesce_float("stop_loss_pct", cfg.stop_loss_pct),
        take_profit_pct=coalesce_float("take_profit_pct", cfg.take_profit_pct),
        volume_spike_thr=coalesce_float("volume_spike_thr", cfg.volume_spike_thr),
        per_symbol_cooldown_sec=coalesce_float("per_symbol_cooldown_sec", cfg.per_symbol_cooldown_sec),
        signal_gap_sec=coalesce_float("signal_gap_sec", cfg.signal_gap_sec),
        confirm_ticks=coalesce_int("confirm_ticks", cfg.confirm_ticks),
        exit_on_special_quote=cfg.exit_on_special_quote,
        block_signs=cfg.block_signs,
        reopen_sign=cfg.reopen_sign,
        disable_minutes_after_special=cfg.disable_minutes_after_special,
        open_delay_sec=cfg.open_delay_sec,
        buyup_mode=coalesce_buyup_mode(cfg.buyup_mode),
        buyup_trail_ticks=max(
            0, coalesce_int("buyup_trail_ticks", cfg.buyup_trail_ticks)
        ),
        volume_min_floor=coalesce_float("volume_min_floor", cfg.volume_min_floor),
        volume_fade_window=max(
            1, coalesce_int("volume_fade_window", cfg.volume_fade_window)
        ),
        volume_fade_tol=max(0.0, coalesce_float("volume_fade_tol", cfg.volume_fade_tol)),
        volume_fade_max_lag=max(
            0, coalesce_int("volume_fade_max_lag", cfg.volume_fade_max_lag)
        ),
        ext_vwap_max_pct=max(
            0.0, coalesce_float("ext_vwap_max_pct", cfg.ext_vwap_max_pct)
        ),
        range_explode_window=max(
            1, coalesce_int("range_explode_window", cfg.range_explode_window)
        ),
        range_explode_k=max(0.0, coalesce_float("range_explode_k", cfg.range_explode_k)),
        range_explode_cooldown_sec=max(
            0.0,
            coalesce_float(
                "range_explode_cooldown_sec", cfg.range_explode_cooldown_sec
            ),
        ),
        pullback_ticks=max(0, coalesce_int("pullback_ticks", cfg.pullback_ticks)),
        pullback_rebreak_ticks=max(
            0, coalesce_int("pullback_rebreak_ticks", cfg.pullback_rebreak_ticks)
        ),
        cooldown_after_stop_sec=max(
            0.0, coalesce_float("cooldown_after_stop_sec", cfg.cooldown_after_stop_sec)
        ),
        chop_box_ticks=max(0, coalesce_int("chop_box_ticks", cfg.chop_box_ticks)),
        chop_silence_sec=max(
            0.0, coalesce_float("chop_silence_sec", cfg.chop_silence_sec)
        ),
    )

def _cleanup_pid() -> None:
    global _singleton_handle, _pidfile_path
    try:
        if _singleton_handle:
            ctypes.windll.kernel32.CloseHandle(_singleton_handle)
    except Exception:
        pass
    try:
        if _pidfile_path and _pidfile_path.exists():
            _pidfile_path.unlink(missing_ok=True)
    except Exception:
        pass


def singleton_guard(tag: str) -> None:
    """Prevent duplicate runner instances via Win32 mutex + PID file.

    Note: Global namespace mutexes expect an interactive desktop token; services
    running under restricted principals may require an alternative guard.
    """
    global _singleton_handle, _pidfile_path
    name = f"Global\\{tag}"
    _singleton_handle = ctypes.windll.kernel32.CreateMutexW(None, False, name)
    if ctypes.GetLastError() == 183:
        print(f"[ERROR] {tag} already running", file=sys.stderr)
        sys.exit(1)
    pid_dir = REPO_ROOT / "runtime" / "pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    _pidfile_path = pid_dir / f"{tag}.pid"
    try:
        _pidfile_path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass
    atexit.register(_cleanup_pid)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        value_str = str(value).strip()
        if not value_str:
            return default
        return float(value_str)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        if isinstance(value, int):
            return value
        return int(float(value))
    except (TypeError, ValueError):
        return default


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def resolve_threshold_path(symbol: str, cli_path: Optional[str]) -> Path:
    if cli_path:
        provided = Path(resolve_path(cli_path))
        if provided.is_file():
            return provided
        raise SystemExit(f"--thr not found: {cli_path}")
    symbol_norm = normalize_symbol(symbol)
    candidate = REPO_ROOT / f"exports/best_thresholds_{symbol_norm}_latest.json"
    if candidate.is_file():
        return candidate
    raise SystemExit(f"threshold json not found: {candidate}")


def parse_flatten_at(value: Optional[str]) -> Optional[dtime]:
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        raise SystemExit(f"Invalid --flatten-at value: {value}")
    hour = safe_int(parts[0])
    minute = safe_int(parts[1])
    if hour is None or minute is None:
        raise SystemExit(f"Invalid --flatten-at value: {value}")
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise SystemExit(f"Invalid --flatten-at value: {value}")
    return dtime(hour=hour, minute=minute)


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip()
    if not s:
        return s
    return s.split(".", 1)[0]


DEFAULT_BLOCK_SIGNS: Tuple[str, ...] = (
    "0000",
    "0102",
    "0103",
    "0107",
    "0108",
    "0109",
    "0116",
    "0117",
    "0118",
    "0119",
    "0120",
)


def _coerce_sign(value: Any) -> str:
    return str(value or "").strip()


BUYUP_SIGNS: Tuple[str, ...] = ("0104", "0120")


def is_buyup(sign: Optional[str]) -> bool:
    return _coerce_sign(sign) in BUYUP_SIGNS


@dataclass(frozen=True)
class RunnerConfig:
    features_db: str
    ops_db: str
    symbols: List[str]
    symbols_original: List[str] = field(default_factory=list)
    poll_interval_sec: float = 1.0
    initial_cash: float = 1_500_000.0
    max_cash_per_trade: float = 1_000_000.0
    fee_rate_bps: float = 0.0
    slippage_ticks: float = 0.0
    tick_size: float = 0.1
    tick_value: float = 1.0
    min_lot: int = 100
    risk_per_trade_pct: float = 0.01
    max_concurrent_positions: int = 1
    daily_loss_limit_pct: float = 0.01
    stats_interval_sec: float = 300.0
    stop_loss_ticks: float = 5.0
    log_path: str = "logs/naut_runner.log"
    timezone: str = "Asia/Tokyo"
    killswitch_check_interval_sec: float = 5.0
    market_window: Optional[str] = None
    stop_loss_pct: float = 0.5
    take_profit_pct: float = 1.0
    volume_spike_thr: float = 1.5
    # Anti-spam / guard rails
    per_symbol_cooldown_sec: float = 30.0
    signal_gap_sec: float = 20.0
    confirm_ticks: int = 3
    exit_on_special_quote: bool = True
    block_signs: Tuple[str, ...] = DEFAULT_BLOCK_SIGNS
    reopen_sign: str = "0101"
    disable_minutes_after_special: float = 15.0
    open_delay_sec: float = 60.0
    buyup_mode: str = "EXIT"
    buyup_trail_ticks: int = 3
    volume_min_floor: float = 1000.0
    volume_fade_window: int = 8
    volume_fade_tol: float = 0.20
    volume_fade_max_lag: int = 2
    ext_vwap_max_pct: float = 0.005
    range_explode_window: int = 12
    range_explode_k: float = 3.0
    range_explode_cooldown_sec: float = 30.0
    pullback_ticks: int = 3
    pullback_rebreak_ticks: int = 1
    cooldown_after_stop_sec: float = 30.0
    chop_box_ticks: int = 4
    chop_silence_sec: float = 30.0
    momentum_quality_min: float = field(default=6.0, metadata={"overridable": False})
    breakout_confirm_bars: int = field(default=3,   metadata={"overridable": False})
    breakout_hold_tolerance: float = 0.004


def _is_special(sign: Any, cfg: RunnerConfig) -> bool:
    value = _coerce_sign(sign)
    if not value:
        return True
    return value in cfg.block_signs


def _is_general(sign: Any, cfg: RunnerConfig) -> bool:
    reopen_sign = _coerce_sign(cfg.reopen_sign)
    if not reopen_sign:
        return False
    return _coerce_sign(sign) == reopen_sign


def _format_sign_code(sign: Any) -> str:
    code = _coerce_sign(sign)
    if len(code) == 4 and code.isdigit():
        return code
    if len(code) >= 4:
        return code[:4]
    if code:
        return code.zfill(4) if code.isdigit() else code.rjust(4, "_")
    return "----"


def _format_epoch_hms(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "--:--:--"


def _log_extra(data_ts):
    try:
        ts = float(data_ts) if data_ts is not None else 0.0
    except (TypeError, ValueError):
        ts = 0.0
    return {"data_ts": ts}


def parse_current_price_time(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    value = str(raw).strip()
    if not value:
        return None
    if " " in value and "T" not in value:
        value = value.replace(" ", "T", 1)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    if len(value) >= 5 and (value[-5] in "+-") and value[-3] != ":":
        value = value[:-2] + ":" + value[-2:]
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone(timedelta(hours=9)))
    return dt.timestamp()

class RunnerClock:
    def __init__(self):
        self.last_ts = 0.0
        self.last_source = "INIT"

    def _apply(self, value: Optional[float], source: str) -> float:
        if value is None:
            return self.last_ts
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return self.last_ts
        if candidate <= 0.0:
            return self.last_ts
        if candidate < self.last_ts:
            candidate = self.last_ts
        self.last_ts = candidate
        self.last_source = source
        _update_latest_data_ts(self.last_ts)
        return self.last_ts

    def update_from_row(self, row, wall_ts=None):
        def _pick_quote_time() -> Optional[float]:
            """
            最良気配時刻を row から取り出して epoch 秒にする。
            優先: max(BidTime, AskTime)。形式は ISO8601 / epoch / HH:MM(:SS) を許容。
            HH:MM(:SS) の場合は fs.t_exec の日付で補完する。
            """
            # 1) 候補取り出し
            cands = []
            for key in ("BidTime", "bid_time", "AskTime", "ask_time"):
                v = row.get(key)
                if v is None or str(v).strip() == "":
                    continue
                s = str(v).strip()

                # 2) ISO8601 or 「日付っぽい」文字列は parse_current_price_time でそのまま
                if any(ch in s for ch in ("T", "-", "Z", "+")):
                    ts = parse_current_price_time(s)
                    if ts is not None:
                        cands.append(ts)
                    continue

                # 3) 数値なら epoch（ms の可能性も）
                try:
                    fv = float(s)
                    if fv > 1e12:  # epoch milliseconds
                        cands.append(fv / 1000.0)
                    else:
                        cands.append(fv)  # epoch seconds
                    continue
                except ValueError:
                    pass

                # 4) HH:MM[:SS] だけのケースは t_exec の日付で合成
                base = safe_float(row.get("t_exec"))
                if base is None:
                    continue
                try:
                    base_dt = datetime.fromtimestamp(float(base))
                    parts = [int(x) for x in s.split(":")]
                    if len(parts) == 2:
                        hh, mm = parts
                        ss = 0
                    else:
                        hh, mm, ss = parts[:3]
                    dt = base_dt.replace(hour=hh, minute=mm, second=ss, microsecond=0)
                    cands.append(dt.timestamp())
                except Exception:
                    continue

            if not cands:
                return None
            # Bid/Ask の両方がある場合はより新しい方（max）を返す
            return max(cands)

        cpt = parse_current_price_time(
            row.get("CurrentPriceTime") or row.get("current_price_time")
        )
        if cpt is not None:
            return self._apply(cpt, "CPT")

        qt = _pick_quote_time()
        if qt is not None:
            return self._apply(qt, "QUOTE")

        for key in ("t_recv", "t_exec"):
            v = safe_float(row.get(key))
            if v is not None:
                return self._apply(v, key.upper())

        return self.last_ts

    def now(self):
        return self.last_ts


def load_runner_config(config_path: Path) -> RunnerConfig:
    payload = load_json_utf8(str(config_path))
    try:
        symbols_raw = payload["symbols"]
    except KeyError as exc:
        raise SystemExit(f"Runner config missing symbols: {config_path}") from exc
    if not isinstance(symbols_raw, list) or not symbols_raw:
        raise SystemExit("Runner config requires non-empty list of symbols")
    symbols_clean = [normalize_symbol(sym) for sym in symbols_raw]
    seen: set[str] = set()
    symbols_norm: List[str] = []
    for sym in symbols_clean:
        if sym and sym not in seen:
            seen.add(sym)
            symbols_norm.append(sym)
    features_db = resolve_path(payload.get("features_db", "naut_market.db"))
    ops_db = resolve_path(payload.get("ops_db", "naut_ops.db"))
    log_path = payload.get("log_path", "logs/naut_runner.log")
    if not Path(log_path).is_absolute():
        log_path = str((REPO_ROOT / log_path).resolve())
    exit_on_special_quote = bool(payload.get("exit_on_special_quote", True))
    raw_block_signs = payload.get("block_signs", DEFAULT_BLOCK_SIGNS)
    if raw_block_signs is None:
        raw_block_signs = DEFAULT_BLOCK_SIGNS
    if isinstance(raw_block_signs, (list, tuple, set)):
        block_source = raw_block_signs
    else:
        block_source = [raw_block_signs]
    block_signs_clean: List[str] = []
    for sign in block_source:
        sign_str = _coerce_sign(sign)
        if sign_str:
            block_signs_clean.append(sign_str)
    block_signs = (
        tuple(block_signs_clean) if block_signs_clean else DEFAULT_BLOCK_SIGNS
    )
    reopen_sign = _coerce_sign(payload.get("reopen_sign", "0101")) or "0101"
    disable_minutes_after_special = float(
        payload.get("disable_minutes_after_special", 15.0)
    )
    open_delay_sec = max(0.0, float(payload.get("open_delay_sec", 60.0)))
    buyup_mode_raw = str(payload.get("buyup_mode", "exit") or "exit").strip().upper()
    if buyup_mode_raw not in {"EXIT", "HOLD", "TRAIL"}:
        buyup_mode_raw = "EXIT"
    buyup_trail_ticks_val = safe_int(payload.get("buyup_trail_ticks"), 3)
    if buyup_trail_ticks_val is None:
        buyup_trail_ticks_val = 3
    buyup_trail_ticks = max(0, int(buyup_trail_ticks_val))
    volume_min_floor = safe_float(payload.get("volume_min_floor"), 1000.0)
    if volume_min_floor is None:
        volume_min_floor = 1000.0
    volume_fade_window = safe_int(payload.get("volume_fade_window"), 8)
    if volume_fade_window is None or volume_fade_window <= 0:
        volume_fade_window = 8
    volume_fade_tol = safe_float(payload.get("volume_fade_tol"), 0.20)
    if volume_fade_tol is None or volume_fade_tol < 0.0:
        volume_fade_tol = 0.20
    volume_fade_max_lag = safe_int(payload.get("volume_fade_max_lag"), 2)
    if volume_fade_max_lag is None or volume_fade_max_lag < 0:
        volume_fade_max_lag = 2
    ext_vwap_max_pct = safe_float(payload.get("ext_vwap_max_pct"), 0.005)
    if ext_vwap_max_pct is None or ext_vwap_max_pct < 0.0:
        ext_vwap_max_pct = 0.005
    range_explode_window = safe_int(payload.get("range_explode_window"), 12)
    if range_explode_window is None or range_explode_window <= 0:
        range_explode_window = 12
    range_explode_k = safe_float(payload.get("range_explode_k"), 3.0)
    if range_explode_k is None or range_explode_k <= 0.0:
        range_explode_k = 3.0
    range_explode_cooldown_sec = safe_float(
        payload.get("range_explode_cooldown_sec"), 30.0
    )
    if range_explode_cooldown_sec is None or range_explode_cooldown_sec < 0.0:
        range_explode_cooldown_sec = 30.0
    pullback_ticks = safe_int(payload.get("pullback_ticks"), 3)
    if pullback_ticks is None or pullback_ticks < 0:
        pullback_ticks = 3
    pullback_rebreak_ticks = safe_int(payload.get("pullback_rebreak_ticks"), 1)
    if pullback_rebreak_ticks is None or pullback_rebreak_ticks < 0:
        pullback_rebreak_ticks = 1
    cooldown_after_stop_sec = safe_float(payload.get("cooldown_after_stop_sec"), 30.0)
    if cooldown_after_stop_sec is None or cooldown_after_stop_sec < 0.0:
        cooldown_after_stop_sec = 30.0
    chop_box_ticks = safe_int(payload.get("chop_box_ticks"), 4)
    if chop_box_ticks is None or chop_box_ticks < 0:
        chop_box_ticks = 4
    chop_silence_sec = safe_float(payload.get("chop_silence_sec"), 60.0)
    if chop_silence_sec is None or chop_silence_sec < 0.0:
        chop_silence_sec = 60.0
    momentum_quality_min = safe_float(payload.get("momentum_quality_min"), 6.0)
    if momentum_quality_min is None or momentum_quality_min < 0.0:
        momentum_quality_min = 6.0
    breakout_confirm_bars = safe_int(payload.get("breakout_confirm_bars"), 3)
    if breakout_confirm_bars is None or breakout_confirm_bars < 0:
        breakout_confirm_bars = 3
    return RunnerConfig(
        features_db=features_db,
        ops_db=ops_db,
        symbols=symbols_norm,
        symbols_original=[str(sym) for sym in symbols_raw],
        poll_interval_sec=float(payload.get("poll_interval_sec", 1.0)),
        initial_cash=float(payload.get("initial_cash", 1_500_000.0)),
        fee_rate_bps=float(payload.get("fee_rate_bps", 0.0)),
        slippage_ticks=float(payload.get("slippage_ticks", 0.0)),
        tick_size=float(payload.get("tick_size", 0.1)),
        tick_value=float(payload.get("tick_value", 1.0)),
        min_lot=int(payload.get("min_lot", 100)),
        risk_per_trade_pct=float(payload.get("risk_per_trade_pct", 0.01)),
        max_concurrent_positions=int(payload.get("max_concurrent_positions", 1)),
        daily_loss_limit_pct=float(payload.get("daily_loss_limit_pct", 0.01)),
        stats_interval_sec=float(payload.get("stats_interval_sec", 300.0)),
        stop_loss_ticks=float(payload.get("stop_loss_ticks", 5.0)),
        log_path=log_path,
        timezone=str(payload.get("timezone", "Asia/Tokyo")),
        killswitch_check_interval_sec=float(
            payload.get("killswitch_check_interval_sec", 5.0)
        ),
        market_window=str(payload.get("market_window", "") or "") or None,
        stop_loss_pct=float(payload.get("stop_loss_pct", 0.5)),
        take_profit_pct=float(payload.get("take_profit_pct", 1.0)),
        volume_spike_thr=float(payload.get("volume_spike_thr", 1.5)),
        per_symbol_cooldown_sec=float(payload.get("per_symbol_cooldown_sec", 30.0)),
        signal_gap_sec=float(payload.get("signal_gap_sec", 20.0)),
        confirm_ticks=int(payload.get("confirm_ticks", 3)),
        exit_on_special_quote=exit_on_special_quote,
        block_signs=block_signs,
        reopen_sign=reopen_sign,
        disable_minutes_after_special=disable_minutes_after_special,
        open_delay_sec=open_delay_sec,
        buyup_mode=buyup_mode_raw,
        buyup_trail_ticks=buyup_trail_ticks,
        volume_min_floor=float(volume_min_floor),
        volume_fade_window=int(volume_fade_window),
        volume_fade_tol=float(volume_fade_tol),
        volume_fade_max_lag=int(volume_fade_max_lag),
        ext_vwap_max_pct=float(ext_vwap_max_pct),
        range_explode_window=int(range_explode_window),
        range_explode_k=float(range_explode_k),
        range_explode_cooldown_sec=float(range_explode_cooldown_sec),
        pullback_ticks=int(pullback_ticks),
        pullback_rebreak_ticks=int(pullback_rebreak_ticks),
        cooldown_after_stop_sec=float(cooldown_after_stop_sec),
        chop_box_ticks=int(chop_box_ticks),
        chop_silence_sec=float(chop_silence_sec),
        momentum_quality_min=float(momentum_quality_min),
        breakout_confirm_bars=int(breakout_confirm_bars),
    )


@dataclass
class ThresholdProfile:
    dataset_id: str = ""
    schema_version: str = ""
    md5: str = ""
    created_at: str = ""
    mode: str = "AUTO"
    score_thr_abs: float = 8.0
    uptick_thr: float = 0.2
    spread_max: float = 2.0
    volume_spike_thr: float = 1.5
    rr_tp_sl: float = 3.0
    cooldown_sec: float = 0.0
    runner_max_hold_sec: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def meta(self) -> Dict[str, str]:
        return {
            "thr_md5": self.md5,
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "mode": self.mode,
        }


def load_threshold_profile(path: Path) -> ThresholdProfile:
    if not path.exists():
        raise SystemExit(f"Threshold JSON not found: {path}")
    computed_md5 = compute_md5(path)
    data = load_json_utf8(str(path))
    json_md5 = str(data.get("md5") or data.get("MD5") or computed_md5)
    if json_md5 and json_md5 != computed_md5:
        logger.warning(
            "Threshold md5 mismatch: file=%s payload=%s", computed_md5, json_md5
        )

    def pick(*keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
            upper = key.upper()
            if upper in data:
                return data[upper]
        return None

    profile = ThresholdProfile(raw=data)
    profile.md5 = computed_md5
    profile.dataset_id = str(pick("dataset_id") or "")
    profile.schema_version = str(pick("schema_version") or "")
    profile.created_at = str(
        pick("created_at") or datetime.now(UTC).isoformat()
    )
    raw_mode = pick("mode")
    profile.mode = str(raw_mode or "AUTO").upper()
    profile.cooldown_sec = safe_float(pick("cooldown_sec"), 0.0) or 0.0
    profile.runner_max_hold_sec = safe_float(
        pick("runner_max_hold_sec"), 0.0
    ) or 0.0
    score_abs = safe_float(pick("score_thr_abs"))
    if score_abs is None:
        fallback_score = safe_float(pick("score_thr"), 0.0) or 0.0
        score_abs = abs(fallback_score)
    profile.score_thr_abs = score_abs
    for key in ("uptick_thr", "spread_max", "volume_spike_thr", "rr_tp_sl"):
        value = safe_float(pick(key))
        if value is not None:
            setattr(profile, key, value)
    if profile.rr_tp_sl <= 0:
        profile.rr_tp_sl = 1.0
    return profile


@dataclass
class PolicyDecision:
    should_enter: bool
    side: Optional[str] = None
    entry_px: float = 0.0
    sl_px: float = 0.0
    tp_px: float = 0.0
    reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


class Policy:
    def __init__(self, profile: ThresholdProfile, config: RunnerConfig):
        self.profile = profile
        self.config = config
        self._vr_win = max(1, int(getattr(config, "volume_fade_window", 8)))
        self._vr_tol = max(0.0, float(getattr(config, "volume_fade_tol", 0.20)))
        self._vr_maxlag = max(0, int(getattr(config, "volume_fade_max_lag", 2)))
        self.ext_vwap_max_pct = max(
            0.0, float(getattr(config, "ext_vwap_max_pct", 0.005))
        )
        self.rx_win = max(1, int(getattr(config, "range_explode_window", 12)))
        self.rx_k = max(0.0, float(getattr(config, "range_explode_k", 2.5)))
        self.rx_cdsec = max(
            0.0, float(getattr(config, "range_explode_cooldown_sec", 30.0))
        )
        self.pb_ticks = max(0, int(getattr(config, "pullback_ticks", 2)))
        self.pb_rebr = max(0, int(getattr(config, "pullback_rebreak_ticks", 1)))
        self.momentum_quality_min = max(
            0.0, float(getattr(config, "momentum_quality_min", 6.0))
        )
        self.breakout_confirm_bars = max(
            0, int(getattr(config, "breakout_confirm_bars", 3))
        )
        self.breakout_hold_tolerance = float(getattr(config, "breakout_hold_tolerance", 0.004))

        self._vr_hist: Dict[str, Deque[Tuple[float, float]]] = {}
        self._ranges: Dict[str, Deque[float]] = {}
        self._rx_block_until: Dict[str, float] = {}
        self._peak_hi: Dict[str, float] = {}
        self._pullback_ready: Dict[str, bool] = {}
        self._volume_hist_simple: Dict[str, Deque[float]] = {}
        self._close_hist: Dict[str, Deque[float]] = {}
        self._uptick_streak: Dict[str, int] = {}
        self._last_close: Dict[str, Optional[float]] = {}
        self._recent_high: Dict[str, float] = {}

    @property
    def cooldown_sec(self) -> float:
        return max(self.profile.cooldown_sec, 0.0)

    @property
    def max_hold_sec(self) -> float:
        return max(self.profile.runner_max_hold_sec, 0.0)

    def _dq(
        self, store: Dict[str, Deque[Any]], key: str, maxlen: int
    ) -> Deque[Any]:
        dq = store.get(key)
        if dq is None or dq.maxlen != maxlen:
            dq = deque(maxlen=max(1, maxlen))
            store[key] = dq
        return dq

    def _push_vrate(self, symbol: str, v_rate: float, data_ts: float) -> Deque[Tuple[float, float]]:
        dq = self._dq(self._vr_hist, symbol, self._vr_win)
        dq.append((float(v_rate), float(data_ts)))
        return dq

    def _fade_blocked(self, dq: Deque[Tuple[float, float]]) -> Tuple[bool, int, float, float]:
        if not dq:
            return False, 0, 0.0, 0.0
        values = [item[0] for item in dq]
        peak = max(values)
        cur = values[-1]
        peak_idx = len(values) - 1 - values[::-1].index(peak)
        lag = len(values) - 1 - peak_idx
        fading = cur < peak * (1.0 - self._vr_tol)
        return fading and lag > 0, lag, cur, peak

    def _volume_value(self, row: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
        for key in keys:
            if key in row:
                value = safe_float(row.get(key))
                if value is not None:
                    return value
        return None

    def _row_timestamp(self, row: Dict[str, Any]) -> float:
        candidates = (
            safe_float(row.get("t_exec")),
            safe_float(row.get("ts")),
            safe_float(row.get("ts_ms")),
        )
        for value in candidates:
            if value is None:
                continue
            if value > 1e12:
                return float(value) / 1000.0
            return float(value)
        return 0.0

    def evaluate(
        self,
        symbol: str,
        row: Dict[str, Any],
        data_ts: Optional[float] = None,
    ) -> PolicyDecision:
        score = self._extract(row, ("score",))
        uptick = self._extract(row, ("uptick_ratio", "uptick", "f1"))
        downtick = self._extract(row, ("downtick_ratio", "downtick", "f2"))
        spread = self._extract(row, ("spread_ticks", "spread", "spread_tick"))
        volume_rate = self._extract(
            row, ("volume_rate", "v_rate", "volume_spike", "volume_ratio")
        )
        vol_now = self._volume_value(row, ("volume", "vol", "v_now", "v"))
        ask_px = self._entry_price(row, SIDE_BUY)
        bid_px = self._entry_price(row, SIDE_SELL)

        context = {
            "symbol": symbol,
            "score": score,
            "uptick_ratio": uptick,
            "downtick_ratio": downtick,
            "spread_ticks": spread,
            "volume_rate": volume_rate,
            "volume": vol_now,
            "ask_px": ask_px,
            "bid_px": bid_px,
        }

        if volume_rate is None:
            ctx = dict(context)
            return PolicyDecision(False, reason="no_volume_rate", context=ctx)
        volume_min_floor = max(
            0.0, float(getattr(self.config, "volume_min_floor", 1000.0))
        )
        if vol_now is not None and vol_now < volume_min_floor:
            ctx = dict(context)
            ctx.update({"volume": vol_now, "volume_min_floor": volume_min_floor})
            return PolicyDecision(False, reason="low_volume", context=ctx)

        supplied_ts = safe_float(data_ts) if data_ts is not None else None
        if supplied_ts is not None and supplied_ts > 0.0:
            data_ts_val = float(supplied_ts)
        else:
            data_ts_val = self._row_timestamp(row)
        price = self._extract(
            row, ("price", "close", "ask1", "ask", "bid1", "bid", "mid")
        )
        if price is None:
            price = ask_px if ask_px is not None else bid_px
        if price is not None:
            context["price"] = price
        close_px = safe_float(row.get("close"))
        if close_px is None:
            close_px = price
        if close_px is not None:
            context["close"] = close_px

        bar_hi = self._extract(row, ("bar_high", "high"))
        if bar_hi is None:
            bar_hi = price if price is not None else ask_px or bid_px
        bar_lo = self._extract(row, ("bar_low", "low"))
        if bar_lo is None:
            bar_lo = price if price is not None else bid_px or ask_px
        tick = max(float(self.config.tick_size), 0.0)

        volume_hist = self._dq(self._volume_hist_simple, symbol, self._vr_win)
        v_rate_calc: Optional[float] = None
        if vol_now is not None:
            if volume_hist:
                avg_volume = sum(volume_hist) / len(volume_hist)
                if avg_volume > 0.0:
                    v_rate_calc = float(vol_now) / avg_volume
            volume_hist.append(float(vol_now))
        close_hist = self._dq(self._close_hist, symbol, self.breakout_confirm_bars)
        last_close_val = self._last_close.get(symbol)
        streak_prev = self._uptick_streak.get(symbol, 0)
        streak_new = streak_prev
        if close_px is not None:
            if last_close_val is not None:
                if close_px > last_close_val:
                    streak_new = streak_prev + 1 if streak_prev > 0 else 1
                elif close_px < last_close_val:
                    streak_new = 0
            else:
                streak_new = 0
            self._last_close[symbol] = close_px
            close_hist.append(float(close_px))
        self._uptick_streak[symbol] = streak_new
        uptick_count = streak_new
        candidate_high = bar_hi if bar_hi is not None else price
        if candidate_high is not None:
            current_high = float(candidate_high)
            stored_high = self._recent_high.get(symbol)
            if stored_high is None or current_high > stored_high:
                self._recent_high[symbol] = current_high
        recent_high = self._recent_high.get(symbol)
        mean_close: Optional[float] = None
        if close_hist and len(close_hist) == self.breakout_confirm_bars:
            mean_close = sum(close_hist) / len(close_hist)

        rng = 0.0
        if bar_hi is not None and bar_lo is not None:
            rng = max(float(bar_hi) - float(bar_lo), 0.0)
        range_dq = self._dq(self._ranges, symbol, self.rx_win)
        range_dq.append(rng)
        median_range: Optional[float] = None
        bid_sign_raw = row.get("BidSign") or row.get("bid_sign")
        ask_sign_raw = row.get("AskSign") or row.get("ask_sign")
        spike_threshold = max(
            float(self.profile.volume_spike_thr),
            float(getattr(self.config, "volume_spike_thr", self.profile.volume_spike_thr)),
        )
        volume_spike_condition = volume_rate is not None and volume_rate >= spike_threshold

        spike_triggered = volume_spike_condition or is_buyup(bid_sign_raw) or is_buyup(
            ask_sign_raw
        )

        dq = self._push_vrate(symbol, volume_rate, data_ts_val)
        if spike_triggered:
            blocked, lag, cur, peak = self._fade_blocked(dq)
            if blocked and lag > self._vr_maxlag:
                tol_pct = self._vr_tol * 100.0
                fade_context = dict(context)
                fade_context.update(
                    {
                        "volume_rate": cur,
                        "volume_peak": peak,
                        "volume_lag": lag,
                        "volume_tol": self._vr_tol,
                    }
                )
                logger.debug(
                    "ENTRY veto by volume fading: v_rate=%.2f peak=%.2f lag=%d tol=%.0f%%",
                    cur, peak, lag, tol_pct,
                    extra=_log_extra(data_ts_val),
                )
                return PolicyDecision(False, reason="volume_fading", context=fade_context)

        if len(range_dq) >= max(3, min(self.rx_win, 5)):
            sorted_ranges = sorted(range_dq)
            median_range = sorted_ranges[len(sorted_ranges) // 2]
            if (
                self.rx_k > 0.0
                and median_range is not None
                and median_range > 0.0
                and rng >= self.rx_k * median_range
                and data_ts_val > 0.0
            ):
                self._rx_block_until[symbol] = data_ts_val + self.rx_cdsec
                logger.debug(
                    "Range explosion detected: rng=%.3f median=%.3f cooldown=%.1fs",
                    rng,
                    median_range,
                    self.rx_cdsec,
                    extra=_log_extra(data_ts_val),
                )
        block_until = self._rx_block_until.get(symbol, 0.0)
        if block_until and (data_ts_val <= 0.0 or data_ts_val < block_until):
            ctx = dict(context)
            ctx.update(
                {
                    "range_rng": rng,
                    "range_median": median_range,
                    "range_block_until": block_until,
                }
            )
            logger.debug(
                "ENTRY veto by range explode cooldown: until=%.3f rng=%.3f",
                block_until,
                rng,
                extra=_log_extra(data_ts_val),
            )
            return PolicyDecision(False, reason="range_explode_cooldown", context=ctx)

        vwap_val = self._extract(row, ("vwap", "ema_fast"))
        if vwap_val is not None:
            context["vwap"] = vwap_val
        if (
            self.ext_vwap_max_pct > 0.0
            and price is not None
            and vwap_val is not None
            and vwap_val > 0.0
        ):
            deviation = price / vwap_val - 1.0
            if deviation > self.ext_vwap_max_pct:
                ctx = dict(context)
                ctx.update({"deviation": deviation})
                logger.debug(
                    "ENTRY veto by VWAP extension: dev=%.4f limit=%.4f",
                    deviation,
                    self.ext_vwap_max_pct,
                    extra=_log_extra(data_ts_val),
                )
                return PolicyDecision(False, reason="too_far_from_vwap", context=ctx)

        peak_ref = self._peak_hi.get(symbol)
        if peak_ref is None or peak_ref <= 0.0:
            if bar_hi is not None:
                self._peak_hi[symbol] = float(bar_hi)
                self._pullback_ready[symbol] = False
            elif price is not None:
                self._peak_hi[symbol] = float(price)
                self._pullback_ready[symbol] = False
            peak_ref = self._peak_hi.get(symbol, 0.0)
        if spike_triggered and (
            self.pb_ticks > 0
            and tick > 0.0
            and price is not None
            and peak_ref is not None
            and peak_ref > 0.0
        ):
            ready = self._pullback_ready.get(symbol, False)
            if price <= peak_ref - self.pb_ticks * tick:
                ready = True
                self._pullback_ready[symbol] = True
            rebreak_threshold = peak_ref + self.pb_rebr * tick
            rebreak = True
            if self.pb_rebr > 0:
                if bar_hi is not None:
                    rebreak = bar_hi >= rebreak_threshold
                else:
                    rebreak = False
            if not ready or (self.pb_rebr > 0 and not rebreak):
                ctx = dict(context)
                ctx.update(
                    {
                        "peak": peak_ref,
                        "price": price,
                        "pullback_ready": ready,
                        "rebreak": rebreak,
                        "rebreak_threshold": rebreak_threshold,
                        "spike_triggered": spike_triggered,
                        "bid_sign": bid_sign_raw,
                        "ask_sign": ask_sign_raw,
                    }
                )
                logger.debug(
                    "ENTRY veto: pullback/rebreak peak=%.3f price=%.3f ready=%s rebreak=%s",
                    peak_ref, price, ready, rebreak,
                    extra=_log_extra(data_ts_val),
                )
                if bar_hi is not None and bar_hi > peak_ref:
                    self._peak_hi[symbol] = float(bar_hi)
                    self._pullback_ready[symbol] = False
                return PolicyDecision(False, reason="need_pullback_rebreak", context=ctx)
            if bar_hi is not None and bar_hi > peak_ref:
                self._peak_hi[symbol] = float(bar_hi)
                self._pullback_ready[symbol] = False

        if bar_hi is not None:
            current_peak = self._peak_hi.get(symbol, 0.0)
            if current_peak <= 0.0 or bar_hi > current_peak:
                self._peak_hi[symbol] = float(bar_hi)
                if self.pb_ticks > 0:
                    self._pullback_ready[symbol] = False

        if v_rate_calc is not None:
            context["momentum_v_rate"] = v_rate_calc
        context["momentum_upticks"] = uptick_count
        if mean_close is not None:
            context["breakout_mean_close"] = mean_close
        if recent_high is not None:
            context["breakout_recent_high"] = recent_high

        momentum_score: Optional[float] = None
        if (
            self.momentum_quality_min > 0.0
            and v_rate_calc is not None
        ):
            momentum_score = v_rate_calc * float(max(uptick_count, 0))
            if momentum_score < self.momentum_quality_min:
                ctx = dict(context)
                ctx.update(
                    {
                        "momentum_v_rate": v_rate_calc,
                        "momentum_upticks": uptick_count,
                        "momentum_score": momentum_score,
                        "momentum_quality_min": self.momentum_quality_min,
                    }
                )
                logger.debug(
                    "ENTRY veto by momentum_quality: v_rate=%.2f upticks=%d score=%.2f thr=%.2f",
                    v_rate_calc,
                    uptick_count,
                    momentum_score,
                    self.momentum_quality_min,
                    extra=_log_extra(data_ts_val),
                )
                return PolicyDecision(False, reason="momentum_quality", context=ctx)
        if momentum_score is not None:
            context["momentum_score"] = momentum_score

        if (
            self.breakout_confirm_bars > 0
            and mean_close is not None
            and recent_high is not None
            and mean_close <= recent_high * (1 - self.breakout_hold_tolerance)
        ):
            ctx = dict(context)
            ctx.update(
                {
                    "breakout_mean_close": mean_close,
                    "breakout_recent_high": recent_high,
                    "breakout_confirm_bars": self.breakout_confirm_bars,
                }
            )
            logger.debug(
                "ENTRY veto by breakout_hold: mean_close=%.3f high=%.3f N=%d",
                mean_close,
                recent_high,
                self.breakout_confirm_bars,
                extra=_log_extra(data_ts_val),
            )
            return PolicyDecision(False, reason="breakout_hold", context=ctx)

        spread_val = float("inf") if spread is None else spread

        stop_pct = max(self.config.stop_loss_pct, 0.0)
        take_pct = self.config.take_profit_pct
        if take_pct <= 0:
            take_pct = stop_pct * max(self.profile.rr_tp_sl, 1.0)
        if stop_pct <= 0 or take_pct <= 0:
            return PolicyDecision(False, reason="stop_pct", context=context)
        stop_pct *= 0.01
        take_pct *= 0.01

        score_thr = max(self.profile.score_thr_abs, 0.0)

        buy_fail: Optional[str] = None
        if ask_px is None:
            buy_fail = "no_quote"
        elif score is None or score < score_thr:
            buy_fail = "score"
        elif uptick is None or uptick < self.profile.uptick_thr:
            buy_fail = "uptick"
        elif spread_val > self.profile.spread_max:
            buy_fail = "spread"
        else:
            sl_px = ask_px * (1.0 - stop_pct)
            tp_px = ask_px * (1.0 + take_pct)
            buy_context = dict(context)
            buy_context.update(
                {
                    "entry_px": ask_px,
                    "sl_px": sl_px,
                    "tp_px": tp_px,
                    "side": SIDE_BUY,
                }
            )
            return PolicyDecision(
                True,
                side=SIDE_BUY,
                entry_px=ask_px,
                sl_px=sl_px,
                tp_px=tp_px,
                reason="signal",
                context=buy_context,
            )

        sell_fail: Optional[str] = None
        if bid_px is None:
            sell_fail = "no_quote"
        elif score is None or score > -score_thr:
            sell_fail = "score"
        elif downtick is None or downtick < self.profile.uptick_thr:
            sell_fail = "downtick"
        elif spread_val > self.profile.spread_max:
            sell_fail = "spread"
        else:
            sl_px = bid_px * (1.0 + stop_pct)
            tp_px = bid_px * (1.0 - take_pct)
            sell_context = dict(context)
            sell_context.update(
                {
                    "entry_px": bid_px,
                    "sl_px": sl_px,
                    "tp_px": tp_px,
                    "side": SIDE_SELL,
                }
            )
            return PolicyDecision(
                True,
                side=SIDE_SELL,
                entry_px=bid_px,
                sl_px=sl_px,
                tp_px=tp_px,
                reason="signal",
                context=sell_context,
            )

        reasons: List[str] = []
        if buy_fail:
            reasons.append(f"BUY={buy_fail}")
        if sell_fail:
            reasons.append(f"SELL={sell_fail}")
        if not reasons:
            reasons.append("filtered")
        fail_context = dict(context)
        fail_context["side"] = None
        return PolicyDecision(
            False,
            side=None,
            reason="|".join(reasons),
            context=fail_context,
        )

    @staticmethod
    def _extract(row: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
        for key in keys:
            if key in row:
                value = safe_float(row.get(key))
                if value is not None:
                    return value
        return None

    def _entry_price(self, row: Dict[str, Any], side: str) -> Optional[float]:
        side_upper = side.upper()
        if side_upper == SIDE_SELL:
            keys = ("bid1", "bid", "bid_px", "bid_price")
        else:
            keys = ("ask1", "ask", "ask_px", "ask_price")
        for key in keys:
            price = safe_float(row.get(key))
            if price is not None:
                return price
        for key in ("price", "last_price", "close", "mid"):
            price = safe_float(row.get(key))
            if price is not None:
                return price
        return None


class FeaturePoller:
    def __init__(self, db_path: str, table: str = "features_stream"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.table = table
        try:
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_symbol_ts ON {self.table}(symbol, t_exec)"
            )
        except Exception:
            logger.debug("FeaturePoller index ensure failed", exc_info=True)

    def fetch_since(
        self, symbol: str, last_ts: float, limit: int = 500
    ) -> List[Dict[str, Any]]:
        query = """
        SELECT fs.*,
            json_extract(r.payload, '$.BidSign') AS BidSign,
            json_extract(r.payload, '$.AskSign') AS AskSign,
            json_extract(r.payload, '$.BidTime') AS BidTime,
            json_extract(r.payload, '$.AskTime') AS AskTime,
            json_extract(r.payload, '$.CurrentPriceTime') AS CurrentPriceTime
        FROM features_stream AS fs
        LEFT JOIN raw_push AS r
        ON r.symbol = fs.symbol
        AND r.t_recv = (
        SELECT t_recv FROM raw_push
        WHERE symbol = fs.symbol AND t_recv <= fs.t_exec
        ORDER BY t_recv DESC LIMIT 1
        )
        WHERE fs.symbol=? AND fs.t_exec>?
        ORDER BY fs.t_exec ASC
        LIMIT ?
        """
        cur = self.conn.execute(query, (symbol, last_ts, limit))
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def latest_timestamps(self, symbols: Iterable[str]) -> Dict[str, float]:
        latest: Dict[str, float] = {}
        for sym in symbols:
            try:
                cur = self.conn.execute(
                    f"SELECT MAX(t_exec) FROM {self.table} WHERE symbol=?",
                    (sym,),
                )
                row = cur.fetchone()
                latest[sym] = float(row[0]) if row and row[0] is not None else 0.0
            except Exception:
                latest[sym] = 0.0
        return latest


class NautRunner:
    def __init__(
        self,
        mode: str,
        config: RunnerConfig,
        policy: Policy,
        broker: Broker,
        ledger: Ledger,
        poller: FeaturePoller,
        trade_logger: TradeLogger,
        flatten_at: Optional[dtime],
        killswitch_path: Path,
        broker_label: str,
        replay_from_start: bool,
    ):
        self.mode = mode.upper()
        self.config = config
        self.policy = policy
        self.broker = broker
        self.ledger = ledger
        self.poller = poller
        self.trade_logger = trade_logger
        self.broker_label = broker_label.lower()
        self.replay_from_start = replay_from_start
        self.symbol = config.symbols[0]
        if len(config.symbols) > 1:
            logger.warning(
                "Multiple symbols configured; using first only: %s (total=%d)",
                self.symbol,
                len(config.symbols),
            )
        if config.symbols_original:
            self.display_symbols = [config.symbols_original[0]]
        else:
            self.display_symbols = [self.symbol]
        self.flatten_at = flatten_at
        self.flatten_triggered = False
        self.killswitch_path = killswitch_path
        self.last_killswitch_check = 0.0
        self.stop_requested = False
        self.cooldown_until: Dict[str, float] = {self.symbol: 0.0}
        self.last_ts: Dict[str, float] = {self.symbol: 0.0}
        self.last_exit_ts: Dict[str, float] = {self.symbol: 0.0}
        self.last_entry_ts: Dict[str, float] = {self.symbol: 0.0}
        self.signal_streak: Dict[str, int] = {self.symbol: 0}
        self.disabled_symbols: Set[str] = set()
        self.entry_gate_reasons: Dict[str, Optional[str]] = {self.symbol: None}
        self.cooldown_after_stop_sec = max(
            0.0, float(getattr(self.config, "cooldown_after_stop_sec", 30.0))
        )
        self.chop_box_ticks = max(0, int(getattr(self.config, "chop_box_ticks", 4)))
        self.chop_silence_sec = max(
            0.0, float(getattr(self.config, "chop_silence_sec", 60.0))
        )
        self._chop_box_center: Optional[float] = None
        self._chop_box_until: float = 0.0
        self._post_stop_box_center: Optional[float] = None
        self._post_stop_box_until: float = 0.0
        self.clock = RunnerClock()
        if self._should_seek_tail():
            latest_map = self.poller.latest_timestamps([self.symbol])
            ts = latest_map.get(self.symbol)
            if ts is not None:
                self.last_ts[self.symbol] = ts
            logger.info(
                "Stream seeked to tail (mode=%s, replay_from_start=%s) symbol=%s@%.3f",
                self.broker_label,
                self.replay_from_start,
                self.symbol,
                self.last_ts[self.symbol],
            )
        else:
            logger.info(
                "Stream starting from head (mode=%s, replay_from_start=%s)",
                self.broker_label,
                self.replay_from_start,
            )
        self.stats: Dict[str, Any] = {
            "polled": 0,
            "signals": 0,
            "entries": 0,
            "exits": 0,
            "wins": 0,
            "losses": 0,
            "pnl_sum": 0.0,
        }
        import sqlite3

        if not hasattr(self, "ops_conn") or self.ops_conn is None:
            self.ops_conn = sqlite3.connect(self.config.ops_db, check_same_thread=False)
            try:
                self.ops_conn.execute("PRAGMA journal_mode=WAL;")
                self.ops_conn.execute("PRAGMA synchronous=NORMAL;")
            except Exception:
                pass
        self._ensure_seen_table()
        self.ops_conn.commit()
        logger.info("runner_seen ready on ops_db=%s", self.config.ops_db)
        self.last_stats_log = 0.0
        self.profile = policy.profile
        self.loss_limit_engaged = False
        self.session_start_ts: float | None = None
        self.session_end_ts: float | None = None

    def _should_seek_tail(self) -> bool:
        if self.broker_label == "live":
            return True
        return not self.replay_from_start

#    def _already_seen(self, symbol: str, ts_ms: int) -> bool:
#        cur = self.ops_conn.execute(
#            "SELECT 1 FROM runner_seen WHERE symbol=? AND profile_md5=? AND ts_ms=?",
#        )
#        return cur.fetchone() is not None
#
#    def _mark_seen(self, symbol: str, ts_ms: int) -> None:
#        self.ops_conn.execute(
#            "INSERT OR IGNORE INTO runner_seen(symbol, profile_md5, ts_ms) VALUES(?,?,?)",
#            (symbol, self.profile.md5, ts_ms),
#        )
#        self.ops_conn.commit()

    def _ensure_seen_table(self):
        try:
            cur = self.ops_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='runner_seen'"
            )
            exists = cur.fetchone() is not None
            if exists:
                cols = {row[1] for row in self.ops_conn.execute("PRAGMA table_info(runner_seen)")}
                expected = {"symbol", "profile_md5", "ts_ms"}
                if cols != expected:
                    # 旧スキーマは落とす（主キーが違うとALTERが難しいため）
                    self.ops_conn.execute("DROP TABLE IF EXISTS runner_seen")
                    self.ops_conn.commit()
            # 無ければ作る
            self.ops_conn.execute("""
                CREATE TABLE IF NOT EXISTS runner_seen(
                    symbol      TEXT NOT NULL,
                    profile_md5 TEXT NOT NULL,
                    ts_ms       INTEGER NOT NULL,
                    PRIMARY KEY(symbol, profile_md5, ts_ms)
                )
            """)
            self.ops_conn.commit()
        except Exception:
            logger.exception("failed to ensure runner_seen table")

    def _already_seen(self, symbol: str, ts_ms: int) -> bool:
        cur = self.ops_conn.execute(
            "SELECT 1 FROM runner_seen WHERE symbol=? AND profile_md5=? AND ts_ms=?",
            (symbol, self.profile.md5, ts_ms),
        )
        return cur.fetchone() is not None

    def _mark_seen(self, symbol: str, ts_ms: int) -> None:
        self.ops_conn.execute(
            "INSERT OR IGNORE INTO runner_seen(symbol, profile_md5, ts_ms) VALUES(?,?,?)",
            (symbol, self.profile.md5, ts_ms),
        )
        self.ops_conn.commit()

    def run(self) -> None:
        logger.info("Runner loop start symbol=%s", ",".join(self.display_symbols))
        self.session_start_ts = time.time()
        symbol = self.symbol
        try:
            while not self.stop_requested:
                data_now = self.clock.now()
                if data_now > 0.0:
                    self._check_flatten(data_now)
                self._check_killswitch()
                self._handle_loss_limit()
                if self.stop_requested:
                    break
                latest_data_ts = data_now
                processed_row = False
                rows = self.poller.fetch_since(symbol, self.last_ts[symbol])
                if rows:
                    for row in rows:
                        data_ts = self.clock.update_from_row(row)
                        latest_data_ts = data_ts
                        processed_row = True
                        row_exec_ts = safe_float(row.get("t_exec"))
                        if row_exec_ts is None:
                            row_exec_ts = data_ts
                        self.last_ts[symbol] = max(self.last_ts[symbol], row_exec_ts)
                        self.stats["polled"] += 1
                        self._mark_positions(symbol, row)
                        self._check_flatten(data_ts)
                        if self.stop_requested:
                            break
                        disable_until = self.cooldown_until.get(symbol, 0.0)
                        if symbol in self.disabled_symbols:
                            if disable_until and data_ts >= disable_until:
                                self.disabled_symbols.discard(symbol)
                            elif disable_until and data_ts < disable_until:
                                continue
                            else:
                                self.disabled_symbols.discard(symbol)
                        fills = self.broker.process_tick(row, data_ts)
                        if fills:
                            self._handle_fills(fills)
                            if self.stop_requested:
                                break
                        self._check_timeouts(symbol, data_ts, row)
                        if self.stop_requested:
                            break
                        self._try_entry(symbol, row, data_ts)
                        if self.stop_requested:
                            break
                if self.stop_requested:
                    break
                if processed_row and latest_data_ts > 0.0:
                    self._maybe_log_stats(latest_data_ts)
                sleep_base = max(float(self.config.poll_interval_sec), 0.0)
                sleep_for = sleep_base if not processed_row else min(sleep_base, 0.1)
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
        finally:
            open_positions = self.ledger.total_positions()
            if open_positions:
                logger.warning("Runner stop with %d open positions", open_positions)

    def shutdown(self) -> None:
        try:
            self.poller.close()
        except Exception:
            pass
        try:
            self.trade_logger.close()
        except Exception:
            pass
    def _handle_fills(self, fills: List[Fill]) -> None:
        for fill in fills:
            summary = self.ledger.register_exit(
                fill.order_id, fill.exit_px, fill.exit_reason, fill.timestamp
            )
            if not summary:
                continue
            self.stats["exits"] += 1
            self.stats["pnl_sum"] += summary.realized_pnl
            if summary.realized_pnl > 0:
                self.stats["wins"] += 1
            elif summary.realized_pnl < 0:
                self.stats["losses"] += 1
            cooldown = max(
                self.policy.cooldown_sec, self.config.per_symbol_cooldown_sec
            )
            base_cooldown_until = fill.timestamp + cooldown
            current_until = self.cooldown_until.get(summary.symbol, 0.0)
            self.cooldown_until[summary.symbol] = max(
                current_until, base_cooldown_until
            )
            self._on_exit(
                summary.symbol,
                summary.exit_reason,
                summary.exit_px if summary.exit_px is not None else fill.exit_px,
                fill.timestamp,
            )
            self.last_exit_ts[summary.symbol] = fill.timestamp
            extra_meta = dict(fill.meta)
            extra_meta.update(
                {
                    "entry_px": summary.entry_px,
                    "exit_px": summary.exit_px,
                    "duration_sec": summary.duration_sec,
                    "realized_pnl": summary.realized_pnl,
                    "fees": summary.fees,
                    "max_unrealized_pnl": summary.max_unrealized,
                    "min_unrealized_pnl": summary.min_unrealized,
                }
            )
            self.trade_logger.log_exit(
                fill.order_id, summary, self.profile.meta, extra_meta
            )
            if summary.exit_reason == "special_quote_exit":
                minutes_from_meta = safe_int(
                    fill.meta.get("disable_minutes_after_special")
                )
                default_minutes = int(
                    max(0.0, float(self.config.disable_minutes_after_special))
                )
                minutes = minutes_from_meta if minutes_from_meta is not None else default_minutes
                self._disable_symbol_temporarily(
                    summary.symbol, minutes, fill.timestamp
                )
            exit_time = _format_epoch_hms(summary.exit_ts)
            logger.info(
                "EXIT %s @%s order=%s side=%s qty=%.0f px=%.3f reason=%s pnl=%.2f md5=%s dataset=%s schema=%s",
                summary.symbol,
                exit_time,
                summary.order_id,
                summary.side,
                summary.qty,
                summary.exit_px,
                summary.exit_reason,
                summary.realized_pnl,
                self.profile.md5,
                self.profile.dataset_id,
                self.profile.schema_version,
                extra=_log_extra(summary.exit_ts),
            )

    def _on_exit(
        self,
        symbol: str,
        reason: str,
        exit_px: Optional[float],
        data_ts: float,
    ) -> None:
        if reason != "stop":
            return
        if self.cooldown_after_stop_sec > 0.0:
            until = data_ts + self.cooldown_after_stop_sec
            previous_until = self.cooldown_until.get(symbol, 0.0)
            self.cooldown_until[symbol] = max(previous_until, until)
            logger.debug(
                "COOLDOWN after STOP: symbol=%s until=%s (%.1fs)",
                symbol,
                _format_epoch_hms(until),
                self.cooldown_after_stop_sec,
                extra=_log_extra(data_ts),
            )
        if (
            self.chop_box_ticks > 0
            and self.chop_silence_sec > 0.0
            and exit_px is not None
        ):
            self._chop_box_center = float(exit_px)
            self._chop_box_until = float(data_ts + self.chop_silence_sec)
            tick_span = self.chop_box_ticks * float(self.config.tick_size)
            logger.debug(
                "CHOP-BOX armed: symbol=%s center=%.3f +/- %d ticks (%.4f) until=%s",
                symbol,
                self._chop_box_center,
                self.chop_box_ticks,
                tick_span,
                _format_epoch_hms(self._chop_box_until),
                extra=_log_extra(data_ts),
            )

    def _disable_symbol_temporarily(
        self, symbol: str, minutes: int, data_ts: float
    ) -> None:
        duration_minutes = max(0, int(minutes))
        if duration_minutes <= 0:
            return
        disable_until = data_ts + duration_minutes * 60
        current_until = self.cooldown_until.get(symbol, 0.0)
        self.cooldown_until[symbol] = max(current_until, disable_until)
        self.disabled_symbols.add(symbol)
        logger.info(
            "DISABLE %s for %d min due to special quote.",
            symbol,
            duration_minutes,
            extra=_log_extra(data_ts),
        )
        try:
            self.watchlist_manager.unregister(symbol)  # type: ignore[attr-defined]
        except Exception:
            pass
    def _mid_price(self, row: Dict[str, Any]) -> Optional[float]:
        price = safe_float(row.get("price"))
        if price is None:
            price = safe_float(row.get("close"))
        if price is not None:
            return price
        bid = safe_float(row.get("bid1")) or safe_float(row.get("bid")) or safe_float(
            row.get("bid_px")
        )
        ask = safe_float(row.get("ask1")) or safe_float(row.get("ask")) or safe_float(
            row.get("ask_px")
        )
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        return ask if ask is not None else bid

    def _mark_positions(self, symbol: str, row: Dict[str, Any]) -> None:
        bid = safe_float(
            row.get("bid1"), safe_float(row.get("bid"), safe_float(row.get("bid_px")))
        )
        ask = safe_float(
            row.get("ask1"), safe_float(row.get("ask"), safe_float(row.get("ask_px")))
        )
        if bid is None and ask is None:
            price = safe_float(row.get("price"), safe_float(row.get("last_price")))
            bid = ask = price
        self.ledger.mark_symbol(symbol, bid, ask)

    def _check_timeouts(
        self, symbol: str, data_ts: float, row: Dict[str, Any]
    ) -> None:
        if self.policy.max_hold_sec <= 0:
            return
        timeout_candidates = [
            position
            for position in list(self.ledger.positions.values())
            if position.symbol == symbol
            and (data_ts - position.opened_at) >= self.policy.max_hold_sec
        ]
        if not timeout_candidates:
            return
        bid = safe_float(
            row.get("bid1"), safe_float(row.get("bid"), safe_float(row.get("bid_px")))
        )
        ask = safe_float(
            row.get("ask1"), safe_float(row.get("ask"), safe_float(row.get("ask_px")))
        )
        fills: List[Fill] = []
        for position in timeout_candidates:
            fill = self.broker.force_exit_order(
                position.order_id, data_ts, "timeout", bid, ask
            )
            if fill:
                fills.append(fill)
        if fills:
            self._handle_fills(fills)

    def _log_entry_gate_once(
        self, symbol: str, reason: str, data_ts: float, detail: str
    ) -> None:
        previous = self.entry_gate_reasons.get(symbol)
        if previous == reason:
            return
        self.entry_gate_reasons[symbol] = reason
        transition = f"{previous or 'none'} -> {reason}"
        event_time = _format_epoch_hms(data_ts)
        logger.debug(
            "ENTRY-GATE %s (%s) @%s: %s",
            transition,
            symbol,
            event_time,
            detail,
            extra=_log_extra(data_ts),
        )

    def _clear_entry_gate(self, symbol: str, data_ts: float) -> None:
        previous = self.entry_gate_reasons.get(symbol)
        if previous is None:
            return
        event_time = _format_epoch_hms(data_ts)
        logger.debug(
            "ENTRY-GATE %s -> none (%s) @%s: restored",
            previous,
            symbol,
            event_time,
            extra=_log_extra(data_ts),
        )
        self.entry_gate_reasons[symbol] = None

    def _try_entry(self, symbol: str, row: Dict[str, Any], data_ts: float) -> None:
        ts_ms_val = safe_int(row.get("ts_ms"))
        if ts_ms_val is None:
            ts_ms_val = int(data_ts * 1000)
        ts_ms = int(ts_ms_val)
        event_time = _format_epoch_hms(data_ts)
        if self._already_seen(symbol, ts_ms):
            logger.debug(
                "ENTRY-GATE seen: %s ts=%s", symbol, ts_ms, extra=_log_extra(data_ts)
            )
            return
        if symbol in self.disabled_symbols:
            self._mark_seen(symbol, ts_ms)
            return
        if self._chop_box_until and data_ts >= self._chop_box_until:
            self._chop_box_center = None
            self._chop_box_until = 0.0
        if self.ledger.loss_limit_hit:
            return
        cooldown_until = self.cooldown_until.get(symbol, 0.0)
        if data_ts < cooldown_until:
            logger.debug(
                "ENTRY veto by COOLDOWN: symbol=%s remaining=%.1fs",
                symbol,
                cooldown_until - data_ts,
                extra=_log_extra(data_ts),
            )
            self._mark_seen(symbol, ts_ms)
            self.signal_streak[symbol] = 0
            return
        if (
            self.chop_box_ticks > 0
            and self.chop_silence_sec > 0.0
            and self._chop_box_center is not None
            and self._chop_box_until > data_ts
        ):
            mid_px = self._mid_price(row)
            if mid_px is not None:
                half_span = self.chop_box_ticks * float(self.config.tick_size)
                box_lo = self._chop_box_center - half_span
                box_hi = self._chop_box_center + half_span
                if box_lo <= mid_px <= box_hi:
                    remaining = max(0.0, self._chop_box_until - data_ts)
                    logger.debug(
                        "ENTRY veto by CHOP-BOX: symbol=%s mid=%.3f in [%.3f, %.3f] (%.1fs left)",
                        symbol,
                        mid_px,
                        box_lo,
                        box_hi,
                        remaining,
                        extra=_log_extra(data_ts),
                    )
                    self._mark_seen(symbol, ts_ms)
                    self.signal_streak[symbol] = 0
                    return
        bid_sign = _coerce_sign(row.get("BidSign") or row.get("bid_sign"))
        ask_sign = _coerce_sign(row.get("AskSign") or row.get("ask_sign"))
        reopen_sign = _coerce_sign(self.config.reopen_sign)
        block_signs = set(self.config.block_signs)
        if bid_sign in block_signs or ask_sign in block_signs:
            detail = f"blocked by quote sign code={_format_sign_code(bid_sign)}/{_format_sign_code(ask_sign)}"
            self._log_entry_gate_once(symbol, "quote_sign_block", data_ts, detail)
            self._mark_seen(symbol, ts_ms)
            self.signal_streak[symbol] = 0
            return
        if reopen_sign and (bid_sign != reopen_sign or ask_sign != reopen_sign):
            detail = (
                f"quote mismatch code={_format_sign_code(bid_sign)}/"
                f"{_format_sign_code(ask_sign)} expected={_format_sign_code(reopen_sign)}"
            )
            self._log_entry_gate_once(
                symbol, "quote_sign_mismatch", data_ts, detail
            )
            self._mark_seen(symbol, ts_ms)
            self.signal_streak[symbol] = 0
            return
        open_delay = max(0.0, float(self.config.open_delay_sec))
        if open_delay > 0:
            market_open_dt = datetime.fromtimestamp(data_ts).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            market_open_ts = market_open_dt.timestamp()
            block_until = market_open_ts + open_delay
            if data_ts < block_until:
                remaining = max(0.0, block_until - data_ts)
                detail = f"open delay remaining={remaining:.1f}s"
                self._log_entry_gate_once(symbol, "open_delay", data_ts, detail)
                self._mark_seen(symbol, ts_ms)
                self.signal_streak[symbol] = 0
                return
        self._clear_entry_gate(symbol, data_ts)
        if not self.ledger.can_open(symbol):
            return
        if self.config.signal_gap_sec > 0:
            last_entry_ts = self.last_entry_ts.get(symbol, 0.0)
            if (
                last_entry_ts
                and (data_ts - last_entry_ts) < self.config.signal_gap_sec
            ):
                return
        decision = self.policy.evaluate(symbol, row, data_ts)
        if not decision.should_enter:
            self.signal_streak[symbol] = 0
            return
        if decision.side == SIDE_SELL:
            self._mark_seen(symbol, ts_ms)
            self.signal_streak[symbol] = 0
            logger.info(
                "ENTRY skipped (SELL disabled for now): %s @%s reason=%s",
                symbol,
                event_time,
                decision.reason,
                extra=_log_extra(data_ts),
            )
            return
        self.stats["signals"] += 1
        self.signal_streak[symbol] = self.signal_streak.get(symbol, 0) + 1
        required = max(1, int(self.config.confirm_ticks))
        if self.signal_streak[symbol] < required:
            return
        qty = self.ledger.compute_order_size(decision.entry_px, decision.sl_px)
        if qty <= 0:
            self._mark_seen(symbol, ts_ms)
            if symbol not in self.disabled_symbols:
                self.disabled_symbols.add(symbol)
                logger.warning(
                    "Disable symbol %s: sizing filtered qty=%.0f (entry_px=%.3f sl=%.3f)",
                    symbol,
                    qty,
                    decision.entry_px,
                    decision.sl_px,
                    extra=_log_extra(data_ts),
                )
            return
        if symbol in self.disabled_symbols:
            self.disabled_symbols.discard(symbol)
        order_meta = {
            "thr_md5": self.profile.md5,
            "dataset_id": self.profile.dataset_id,
            "schema_version": self.profile.schema_version,
            "mode": self.profile.mode,
            "stop_loss_pct": self.config.stop_loss_pct,
            "take_profit_pct": self.config.take_profit_pct,
            "score": decision.context.get("score"),
            "uptick_ratio": decision.context.get("uptick_ratio"),
            "downtick_ratio": decision.context.get("downtick_ratio"),
            "spread_ticks": decision.context.get("spread_ticks"),
            "volume_rate": decision.context.get("volume_rate"),
            "entry_px": decision.entry_px,
            "sl_px": decision.sl_px,
            "tp_px": decision.tp_px,
            "side": SIDE_BUY,
        }
        order_id = self.broker.place_ifdoco(
            symbol,
            SIDE_BUY,
            qty,
            decision.entry_px,
            decision.sl_px,
            decision.tp_px,
            order_meta,
            data_ts,
        )
        self.ledger.register_entry(
            order_id,
            symbol,
            SIDE_BUY,
            qty,
            decision.entry_px,
            decision.sl_px,
            decision.tp_px,
            data_ts,
            order_meta,
        )
        self.trade_logger.log_entry(
            order_id,
            data_ts,
            symbol,
            SIDE_BUY,
            qty,
            decision.entry_px,
            decision.reason,
            self.profile.meta,
            order_meta,
        )
        self._mark_seen(symbol, ts_ms)
        self.stats["entries"] += 1
        self.last_entry_ts[symbol] = data_ts
        self.signal_streak[symbol] = 0
        logger.info(
            "ENTRY %s @%s order=%s side=%s qty=%.0f entry=%.3f sl=%.3f tp=%.3f score=%s md5=%s dataset=%s schema=%s",
            symbol,
            event_time,
            order_id,
            SIDE_BUY,
            qty,
            decision.entry_px,
            decision.sl_px,
            decision.tp_px,
            decision.context.get("score"),
            self.profile.md5,
            self.profile.dataset_id,
            self.profile.schema_version,
            extra=_log_extra(data_ts),
        )

    def _check_flatten(self, data_ts: float) -> None:
        if (
            self.flatten_at is None
            or self.flatten_triggered
            or data_ts <= 0.0
        ):
            return
        current_time = datetime.fromtimestamp(data_ts).time()
        if current_time >= self.flatten_at:
            logger.warning(
                "Flatten triggered at %s",
                current_time.strftime("%H:%M"),
                extra=_log_extra(data_ts),
            )
            self._flatten("flatten", data_ts)
            self.flatten_triggered = True
            self.stop_requested = True

    def _check_killswitch(self) -> None:
        current_ts = self.clock.now()
        interval = max(float(self.config.killswitch_check_interval_sec), 0.0)
        if current_ts > 0.0:
            if (
                self.last_killswitch_check > 0.0
                and current_ts - self.last_killswitch_check < interval
            ):
                return
            self.last_killswitch_check = current_ts
        else:
            self.last_killswitch_check = 0.0
        if self.killswitch_path.exists():
            data_ts = current_ts if current_ts > 0.0 else _get_latest_data_ts()
            logger.error(
                "Killswitch detected: %s",
                self.killswitch_path,
                extra=_log_extra(data_ts),
            )
            self._flatten("killswitch", data_ts)
            self.stop_requested = True

    def _handle_loss_limit(self) -> None:
        if self.ledger.loss_limit_hit and not self.loss_limit_engaged:
            self.loss_limit_engaged = True
            drawdown_pct = 0.0
            if self.ledger.initial_cash > 0:
                drawdown_pct = (
                    (self.ledger.initial_cash - self.ledger.equity)
                    / self.ledger.initial_cash
                    * 100
                )
            data_ts = self.clock.now()
            logger.error(
                "Daily loss limit hit: equity=%.2f initial=%.2f drawdown=%.2f%% -> flatten",
                self.ledger.equity,
                self.ledger.initial_cash,
                drawdown_pct,
                extra=_log_extra(data_ts),
            )
            self._flatten("loss_limit", data_ts)
            self.stop_requested = True

    def _flatten(self, reason: str, data_ts: Optional[float] = None) -> None:
        ts_val = safe_float(data_ts)
        if ts_val is None or ts_val <= 0.0:
            ts_val = self.clock.now()
        if ts_val is None or ts_val <= 0.0:
            ts_val = _get_latest_data_ts()
        if ts_val is None or ts_val <= 0.0:
            ts_val = self.clock.last_ts
        if ts_val is None:
            ts_val = 0.0
        if ts_val > self.clock.last_ts:
            self.clock.last_ts = ts_val
        _update_latest_data_ts(self.clock.last_ts)
        fills = self.broker.force_exit_all(reason, ts_val)
        if fills:
            self._handle_fills(fills)
        else:
            logger.warning(
                "Flatten requested (%s) but broker returned no fills",
                reason,
                extra=_log_extra(ts_val),
            )

    def _maybe_log_stats(self, data_ts: float) -> None:
        if data_ts <= 0.0:
            return
        if self.last_stats_log <= 0.0:
            self.last_stats_log = data_ts
            return
        if data_ts - self.last_stats_log < self.config.stats_interval_sec:
            return
        exits = self.stats["exits"]
        win_rate = (self.stats["wins"] / exits) if exits else 0.0
        ev_est = (self.stats["pnl_sum"] / exits) if exits else 0.0
        drawdown_pct = 0.0
        if self.ledger.peak_equity > 0:
            drawdown_pct = (self.ledger.drawdown / self.ledger.peak_equity) * 100.0
        logger.info(
            "stats @%s polled=%d signals=%d entries=%d exits=%d win_rate=%.2f ev_est=%.2f equity=%.2f drawdown=%.2f drawdown_pct=%.2f%%",
            _format_epoch_hms(data_ts),
            self.stats["polled"],
            self.stats["signals"],
            self.stats["entries"],
            exits,
            win_rate,
            ev_est,
            self.ledger.equity,
            self.ledger.drawdown,
            drawdown_pct,
            extra=_log_extra(data_ts),
        )
        self.stats = {
            "polled": 0,
            "signals": 0,
            "entries": 0,
            "exits": 0,
            "wins": 0,
            "losses": 0,
            "pnl_sum": 0.0,
        }
        self.last_stats_log = data_ts

    def show_summary(self, start_ts: float | None = None, end_ts: float | None = None):
        """指定した時間範囲だけを集計して表示（paper_pairs テーブル対応）"""
        import sqlite3
        from statistics import mean

        db_path = Path(self.config.ops_db)
        if not db_path.exists():
            logger.warning(f"DB not found: {db_path}")
            return

        # 範囲デフォルト（念のため）
        if start_ts is None:
            start_ts = 0.0
        if end_ts is None:
            end_ts = time.time()
        start_ms = int(start_ts * 1000)
        end_ms   = int(end_ts * 1000)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # 決済済みトレード（実現損益あり）のみ集計
        c.execute(
            """
            SELECT symbol, side, realized_pnl, exit_reason
            FROM paper_pairs
            WHERE realized_pnl IS NOT NULL
            AND opened_at >= ?
            AND closed_at <= ?
            """,
            (start_ms, end_ms),
        )
        rows = c.fetchall()

        # 期間内にオープンして未決済の件数（参考）
        c.execute(
            """
            SELECT COUNT(1)
            FROM paper_pairs
            WHERE realized_pnl IS NULL
            AND opened_at >= ?
            AND (closed_at IS NULL OR closed_at > ?)
            """,
            (start_ms, end_ms),
        )
        open_cnt = c.fetchone()[0] or 0

        conn.close()

        # 集計
        if not rows:
            logger.info("=" * 48)
            logger.info("📊 Session Summary (window only)")
            logger.info(f"Window: [{start_ts:.0f} .. {end_ts:.0f}]")
            logger.info("No closed trades in the window.")
            if open_cnt:
                logger.info(f"Open (unclosed) trades in window: {open_cnt}")
            logger.info("=" * 48)
            return

        n_trades = len(rows)
        wins   = [p for (_sym, _side, p, _r) in rows if p > 0]
        losses = [p for (_sym, _side, p, _r) in rows if p <= 0]
        ev = mean([p for (_sym, _side, p, _r) in rows])
        win_rate = (len(wins) / n_trades * 100.0) if n_trades else 0.0

        by_symbol: dict[str, list[float]] = {}
        for sym, _side, pnl, _r in rows:
            by_symbol.setdefault(sym, []).append(pnl)

        logger.info("=" * 48)
        logger.info("📊 Session Summary (window only)")
        logger.info(f"Window: [{start_ts:.0f} .. {end_ts:.0f}]")
        logger.info(f"Total trades : {n_trades}")
        logger.info(f"Win rate     : {win_rate:.1f}%")
        logger.info(f"Average EV   : {ev:.2f}")
        if wins and losses:
            from statistics import mean as _mean
            logger.info(f"Avg win/loss : {_mean(wins):.2f} / {_mean(losses):.2f}")
        for sym, plist in by_symbol.items():
            from statistics import mean as _mean
            logger.info(f"  {sym}: {len(plist)} trades, EV={_mean(plist):.2f}")
        if open_cnt:
            logger.info(f"Open (unclosed) trades in window: {open_cnt}")
        logger.info("=" * 48)


def configure_logging(log_path: str, verbose: bool) -> None:
    ensure_parent_dir(log_path)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(message)s"
    formatter = DataTimeFormatter(fmt) 
    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[fh, sh], force=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Naut Runner IFDOCO executor")
    parser.add_argument(
        "--mode",
        choices=["AUTO", SIDE_BUY, SIDE_SELL],
        default="AUTO",
        help="Execution side: AUTO evaluates both BUY/SELL (SELL entries skipped).",
    )
    parser.add_argument(
        "--thr",
        required=False,
        default=None,
        help="Optional path to best_thresholds JSON; auto-resolves when omitted.",
    )
    parser.add_argument("--broker", choices=["paper", "live"], default="paper")
    parser.add_argument(
        "--dry-run",
        type=int,
        choices=[0, 1],
        default=1,
        help="Live only: 0=live orders, 1=dry-run",
    )
    parser.add_argument("--config", required=True, help="Runner config JSON")
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--replay-from-start",
        action="store_true",
        help="In paper mode, read features_stream from the beginning instead of tail.",
    )
    parser.add_argument(
        "--flatten-at", dest="flatten_at", help="HH:MM local time to flatten positions"
    )
    parser.add_argument(
        "--killswitch", default=str(REPO_ROOT / "runtime" / "stop.flag")
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="",
        help="Optional policy.json path to override sizing/risk",
    )
    args = parser.parse_args()

    mode_label = (args.mode or "AUTO").lower()
    singleton_guard(f"naut_runner_{mode_label}_{args.broker}")

    config_path = Path(resolve_path(args.config))
    runner_config = load_runner_config(config_path)
    active_symbol = runner_config.symbols[0]
    threshold_path = resolve_threshold_path(active_symbol, args.thr)
    policy_dict = _load_json_optional(args.policy)
    if policy_dict:
        runner_config = _apply_policy_overrides(runner_config, policy_dict)
        logger.info(
            "POLICY loaded: max_cash_per_trade=%.0f risk_per_trade_pct=%.4f min_lot=%d daily_loss_limit_pct=%.4f initial_cash=%.0f",
            runner_config.max_cash_per_trade,
            runner_config.risk_per_trade_pct,
            runner_config.min_lot,
            runner_config.daily_loss_limit_pct,
            runner_config.initial_cash,
        )
    else:
        logger.info(
            "POLICY not provided. Using built-in defaults: max_cash_per_trade=%.0f risk_per_trade_pct=%.4f min_lot=%d daily_loss_limit_pct=%.4f initial_cash=%.0f",
            runner_config.max_cash_per_trade,
            runner_config.risk_per_trade_pct,
            runner_config.min_lot,
            runner_config.daily_loss_limit_pct,
            runner_config.initial_cash,
        )
    configure_logging(runner_config.log_path, bool(args.verbose))
    if args.broker == "paper" and runner_config.market_window:
        logger.info(
            "Ignoring market_window config in paper mode: %s",
            runner_config.market_window,
        )
    elif runner_config.market_window:
        logger.info(
            "market_window configured=%s (not enforced by current runner)",
            runner_config.market_window,
        )
    if args.mode.upper() == SIDE_SELL:
        logger.warning("SELL execution disabled: runner will only log SELL signals.")

    threshold_profile = load_threshold_profile(threshold_path)
    logger.info("Threshold loaded for %s: %s", active_symbol, threshold_path)

    flatten_at = parse_flatten_at(args.flatten_at)
    killswitch_path = Path(resolve_path(args.killswitch))

    poller = FeaturePoller(runner_config.features_db)
    ledger = Ledger(runner_config)
    policy = Policy(threshold_profile, runner_config)
    trade_logger = TradeLogger(runner_config.ops_db)

    broker: Broker
    if args.broker == "paper":
        broker = PaperBroker(runner_config)
    else:
        broker = LiveBroker(runner_config, dry_run=bool(args.dry_run))

    runner = NautRunner(
        mode=args.mode,
        config=runner_config,
        policy=policy,
        broker=broker,
        ledger=ledger,
        poller=poller,
        trade_logger=trade_logger,
        flatten_at=flatten_at,
        killswitch_path=killswitch_path,
        broker_label=args.broker,
        replay_from_start=bool(args.replay_from_start),
    )

    logger.info(
        "Runner bootstrap mode=%s broker=%s dry_run=%s dataset_id=%s schema=%s md5=%s cooldown=%.1f max_hold=%.1f rr_tp_sl=%.2f",
        args.mode,
        args.broker,
        args.dry_run,
        threshold_profile.dataset_id,
        threshold_profile.schema_version,
        threshold_profile.md5,
        policy.cooldown_sec,
        policy.max_hold_sec,
        threshold_profile.rr_tp_sl,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("Runner stop requested by KeyboardInterrupt")
        runner.session_end_ts = time.time()
    finally:
        if runner.session_end_ts is None:
            runner.session_end_ts = time.time()
        runner.show_summary(start_ts=runner.session_start_ts, end_ts=runner.session_end_ts)
        runner.shutdown()


@dataclass
class Position:
    order_id: str
    symbol: str
    side: str
    qty: float
    entry_px: float
    sl_px: float
    tp_px: float
    opened_at: float
    meta: Dict[str, Any]
    unrealized: float = 0.0
    max_unrealized: float = 0.0
    min_unrealized: float = 0.0
    last_mark_px: float = 0.0


@dataclass
class ExitSummary:
    order_id: str
    symbol: str
    side: str
    qty: float
    entry_px: float
    exit_px: float
    exit_reason: str
    exit_ts: float
    realized_pnl: float
    fees: float
    max_unrealized: float
    min_unrealized: float
    duration_sec: float


class Ledger:
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.initial_cash = config.initial_cash
        self.cash = config.initial_cash
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.exposure = 0.0
        self.positions: Dict[str, Position] = {}
        self.positions_by_symbol: Dict[str, str] = {}
        self.peak_equity = self.initial_cash
        self.drawdown = 0.0
        self.loss_limit_hit = False

    @property
    def equity(self) -> float:
        return self.cash + self.unrealized_pnl

    def total_positions(self) -> int:
        return len(self.positions)

    def can_open(self, symbol: str) -> bool:
        if symbol in self.positions_by_symbol:
            return False
        if self.total_positions() >= self.config.max_concurrent_positions:
            return False
        return True

    def compute_order_size(self, entry_px: float, sl_px: float) -> int:
        risk_per_unit = abs(entry_px - sl_px) * self.config.tick_value
        if risk_per_unit <= 0:
            return 0

        equity = max(self.equity, 0.0)

        # --- リスク％での上限（従来通り）
        risk_cash = (
            equity * self.config.risk_per_trade_pct
            if self.config.risk_per_trade_pct > 0
            else float("inf")
        )
        qty_risk = risk_cash / max(risk_per_unit, 1e-9)

        # --- 現金上限（100万円デフォルト）
        cash_cap = (
            float(self.config.max_cash_per_trade)
            if getattr(self.config, "max_cash_per_trade", 0) > 0
            else float("inf")
        )
        qty_cash = cash_cap / max(entry_px, 1e-9)

        # --- 最小制約（必要ならブローカー上限もここに追加）
        raw_qty = min(qty_risk, qty_cash)

        # --- ロット丸め（既存ロジック踏襲）
        min_lot = max(1, self.config.min_lot)
        lots = math.floor(raw_qty / min_lot)
        qty = lots * min_lot
        if qty <= 0 and raw_qty >= min_lot:
            qty = min_lot
        return int(max(0, qty))

    def register_entry(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        entry_px: float,
        sl_px: float,
        tp_px: float,
        opened_at: float,
        meta: Dict[str, Any],
    ) -> Position:
        position = Position(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            entry_px=entry_px,
            sl_px=sl_px,
            tp_px=tp_px,
            opened_at=opened_at,
            meta=dict(meta),
        )
        self.positions[order_id] = position
        self.positions_by_symbol[symbol] = order_id
        self.exposure += abs(entry_px * qty)
        self._recalc_unrealized()
        self._update_equity()
        return position

    def mark_symbol(
        self, symbol: str, bid: Optional[float], ask: Optional[float]
    ) -> None:
        for position in list(self.positions.values()):
            if position.symbol != symbol:
                continue
            mark_px = bid if position.side == SIDE_BUY else ask
            self.mark(position.order_id, mark_px)

    def mark(self, order_id: str, mark_px: Optional[float]) -> None:
        if mark_px is None:
            return
        position = self.positions.get(order_id)
        if not position:
            return
        position.last_mark_px = mark_px
        diff = (
            (mark_px - position.entry_px)
            if position.side == SIDE_BUY
            else (position.entry_px - mark_px)
        )
        pnl = diff * position.qty * self.config.tick_value
        position.unrealized = pnl
        position.max_unrealized = max(position.max_unrealized, pnl)
        position.min_unrealized = min(position.min_unrealized, pnl)
        self._recalc_unrealized()
        self._update_equity()

    def register_exit(
        self, order_id: str, exit_px: float, reason: str, exit_ts: float
    ) -> Optional[ExitSummary]:
        position = self.positions.pop(order_id, None)
        if not position:
            return None
        self.positions_by_symbol.pop(position.symbol, None)
        diff = (
            (exit_px - position.entry_px)
            if position.side == SIDE_BUY
            else (position.entry_px - exit_px)
        )
        gross_realized = diff * position.qty * self.config.tick_value
        fees = self._calc_fees(position.entry_px, exit_px, position.qty)
        net_realized = gross_realized - fees
        self.cash += net_realized
        self.realized_pnl += net_realized
        self.exposure = max(0.0, self.exposure - abs(position.entry_px * position.qty))
        self.unrealized_pnl -= position.unrealized
        summary = ExitSummary(
            order_id=order_id,
            symbol=position.symbol,
            side=position.side,
            qty=position.qty,
            entry_px=position.entry_px,
            exit_px=exit_px,
            exit_reason=reason,
            exit_ts=exit_ts,
            realized_pnl=net_realized,
            fees=fees,
            max_unrealized=position.max_unrealized,
            min_unrealized=position.min_unrealized,
            duration_sec=max(0.0, exit_ts - position.opened_at),
        )
        self._recalc_unrealized()
        self._update_equity()
        return summary

    def _calc_fees(self, entry_px: float, exit_px: float, qty: float) -> float:
        bps = self.config.fee_rate_bps / 10000.0
        if bps <= 0:
            return 0.0
        notional = (abs(entry_px) + abs(exit_px)) * qty
        return notional * bps

    def _recalc_unrealized(self) -> None:
        self.unrealized_pnl = sum(
            position.unrealized for position in self.positions.values()
        )

    def _update_equity(self) -> None:
        equity = self.equity
        self.peak_equity = max(self.peak_equity, equity)
        self.drawdown = max(0.0, self.peak_equity - equity)
        loss_limit_amount = self.initial_cash * self.config.daily_loss_limit_pct
        if loss_limit_amount > 0:
            drawdown_from_initial = max(0.0, self.initial_cash - equity)
            if drawdown_from_initial >= loss_limit_amount:
                self.loss_limit_hit = True


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: str
    qty: float
    entry_px: float
    sl_px: float
    tp_px: float
    opened_at: float
    meta: Dict[str, Any]
    last_bid: Optional[float] = None
    last_ask: Optional[float] = None


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    qty: float
    exit_px: float
    exit_reason: str
    timestamp: float
    meta: Dict[str, Any]


class Broker:
    def place_ifdoco(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_px: float,
        sl_px: float,
        tp_px: float,
        meta: Dict[str, Any],
        opened_at: float,
    ) -> str:
        raise NotImplementedError

    def process_tick(self, row: Dict[str, Any], timestamp: float) -> List[Fill]:
        return []

    def force_exit_order(
        self,
        order_id: str,
        timestamp: float,
        reason: str,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> Optional[Fill]:
        return None

    def force_exit_all(self, reason: str, timestamp: float) -> List[Fill]:
        return []

    def list_open_orders(self) -> List[PaperOrder]:
        return []


class PaperBroker(Broker):
    def __init__(self, config: RunnerConfig):
        self.config = config
        self._orders: Dict[str, PaperOrder] = {}
        self._sequence = 0

    def place_ifdoco(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_px: float,
        sl_px: float,
        tp_px: float,
        meta: Dict[str, Any],
        opened_at: float,
    ) -> str:
        self._sequence += 1
        order_id = f"PB-{self._sequence:06d}"
        self._orders[order_id] = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            entry_px=entry_px,
            sl_px=sl_px,
            tp_px=tp_px,
            opened_at=opened_at,
            meta=dict(meta),
        )
        logger.debug(
            "Paper IFDOCO placed order_id=%s symbol=%s side=%s qty=%.0f entry=%.3f sl=%.3f tp=%.3f",
            order_id,
            symbol,
            side,
            qty,
            entry_px,
            sl_px,
            tp_px,
            extra=_log_extra(opened_at),
        )
        return order_id

    def process_tick(self, row: Dict[str, Any], timestamp: float) -> List[Fill]:
        symbol = row.get("symbol")
        if not symbol:
            return []
        bid = safe_float(
            row.get("bid1"), safe_float(row.get("bid"), safe_float(row.get("bid_px")))
        )
        ask = safe_float(
            row.get("ask1"), safe_float(row.get("ask"), safe_float(row.get("ask_px")))
        )
        bid_sign = _coerce_sign(row.get("BidSign") or row.get("bid_sign"))
        ask_sign = _coerce_sign(row.get("AskSign") or row.get("ask_sign"))
        buyup_mode_cfg = str(
            getattr(self.config, "buyup_mode", "EXIT") or "EXIT"
        ).strip().lower()
        trail_ticks_cfg = safe_int(
            getattr(self.config, "buyup_trail_ticks", 3), 3
        )
        if trail_ticks_cfg is None:
            trail_ticks_cfg = 3
        buyup_trail_ticks = max(0, int(trail_ticks_cfg))
        fills: List[Fill] = []
        for order in list(self._orders.values()):
            if order.symbol != symbol:
                continue
            if bid is not None:
                order.last_bid = bid
            if ask is not None:
                order.last_ask = ask
            if self.config.exit_on_special_quote:
                if _is_special(bid_sign, self.config) or _is_special(
                    ask_sign, self.config
                ):
                    if (
                        order.side == SIDE_BUY
                        and buyup_mode_cfg in ("hold", "trail")
                        and (is_buyup(bid_sign) or is_buyup(ask_sign))
                    ):
                        logger.debug(
                            "buyup_mode=%s: hold position on buy-up sign bid=%s ask=%s",
                            buyup_mode_cfg,
                            bid_sign or "-",
                            ask_sign or "-",
                            extra=_log_extra(timestamp),
                        )
                        order.meta["pending_special_exit"] = False
                        order.meta["frozen"] = False
                        if buyup_mode_cfg == "trail":
                            basis = (
                                order.last_bid
                                if order.side == SIDE_BUY
                                else order.last_ask
                            )
                            if basis is None:
                                basis = bid if order.side == SIDE_BUY else ask
                            if basis is None:
                                basis = order.entry_px
                            if basis is not None:
                                tick_size = max(float(self.config.tick_size), 0.0)
                                if tick_size <= 0.0:
                                    new_sl = basis
                                else:
                                    new_sl = basis - buyup_trail_ticks * tick_size
                                if order.sl_px is None or new_sl > order.sl_px:
                                    prev_sl = order.sl_px
                                    order.sl_px = new_sl
                                    logger.debug(
                                        "trailing tightened: stop_loss_px=%.3f (ticks=%d, basis=%.3f, prev_sl=%.3f)",
                                        order.sl_px, buyup_trail_ticks, basis, prev_sl, extra=_log_extra(timestamp)
                                    )
                        continue
                    if not order.meta.get("pending_special_exit"):
                        logger.debug(
                            "special_quote detected: reserve exit for %s bid=%s ask=%s",
                            order.symbol,
                            bid_sign or "-",
                            ask_sign or "-",
                            extra=_log_extra(timestamp),
                        )
                    order.meta["pending_special_exit"] = True
                    order.meta["frozen"] = True
                    continue
                if (
                    order.meta.get("pending_special_exit")
                    and _is_general(bid_sign, self.config)
                    and _is_general(ask_sign, self.config)
                ):
                    basis = order.last_bid if order.side == SIDE_BUY else order.last_ask
                    if basis is None:
                        logger.debug(
                            "special_quote exit pending: missing quote for %s",
                            order.symbol,
                            extra=_log_extra(timestamp),
                        )
                        continue
                    minutes = int(
                        max(0.0, float(self.config.disable_minutes_after_special))
                    )
                    order.meta["pending_special_exit"] = False
                    order.meta["special_quote_exit"] = True
                    order.meta["disable_minutes_after_special"] = minutes
                    fill = self._finalize(order, basis, "special_quote_exit", timestamp)
                    logger.debug(
                        "special_quote exit filled: %s px=%.3f",
                        order.symbol,
                        basis,
                        extra=_log_extra(timestamp),
                    )
                    fills.append(fill)
                    continue
            exit_reason = None
            exit_px = None
            if order.side == SIDE_BUY:
                if order.last_bid is not None and order.last_bid <= order.sl_px:
                    exit_reason = "stop"
                    exit_px = order.last_bid
                elif order.last_bid is not None and order.last_bid >= order.tp_px:
                    exit_reason = "target"
                    exit_px = order.last_bid
            else:
                if order.last_ask is not None and order.last_ask >= order.sl_px:
                    exit_reason = "stop"
                    exit_px = order.last_ask
                elif order.last_ask is not None and order.last_ask <= order.tp_px:
                    exit_reason = "target"
                    exit_px = order.last_ask
            if exit_reason and exit_px is not None:
                fills.append(self._finalize(order, exit_px, exit_reason, timestamp))
        return fills

    def force_exit_order(
        self,
        order_id: str,
        timestamp: float,
        reason: str,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> Optional[Fill]:
        order = self._orders.get(order_id)
        if not order:
            return None
        if bid is not None:
            order.last_bid = bid
        if ask is not None:
            order.last_ask = ask
        exit_px = order.last_bid if order.side == SIDE_BUY else order.last_ask
        if exit_px is None:
            return None
        return self._finalize(order, exit_px, reason, timestamp)

    def force_exit_all(self, reason: str, timestamp: float) -> List[Fill]:
        fills: List[Fill] = []
        for order in list(self._orders.values()):
            fill = self.force_exit_order(order.order_id, timestamp, reason)
            if fill:
                fills.append(fill)
        return fills

    def list_open_orders(self) -> List[PaperOrder]:
        return list(self._orders.values())

    def _finalize(
        self, order: PaperOrder, exit_px: float, reason: str, timestamp: float
    ) -> Fill:
        self._orders.pop(order.order_id, None)
        meta = dict(order.meta)
        meta["exit_reason"] = reason
        meta["exit_ts"] = timestamp
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            exit_px=exit_px,
            exit_reason=reason,
            timestamp=timestamp,
            meta=meta,
        )


class LiveBroker(Broker):
    def __init__(self, config: RunnerConfig, dry_run: bool):
        self.config = config
        self.dry_run = dry_run
        self._sequence = 0
        self._paper_delegate = PaperBroker(config) if dry_run else None

    def place_ifdoco(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_px: float,
        sl_px: float,
        tp_px: float,
        meta: Dict[str, Any],
        opened_at: float,
    ) -> str:
        if self.dry_run and self._paper_delegate:
            logger.info(
                "[LIVE-DRYRUN] place_ifdoco symbol=%s side=%s qty=%.0f entry=%.3f sl=%.3f tp=%.3f meta=%s",
                symbol,
                side,
                qty,
                entry_px,
                sl_px,
                tp_px,
                json.dumps(meta, ensure_ascii=False, sort_keys=True),
                extra=_log_extra(opened_at),
            )
            return self._paper_delegate.place_ifdoco(
                symbol, side, qty, entry_px, sl_px, tp_px, meta, opened_at
            )
        # TODO: Wire live execution via market entry + immediate stop/limit legs, with retry/cancel/status
        # flows and request throttling aligned to broker API constraints.
        self._sequence += 1
        order_id = f"LV-{self._sequence:06d}"
        logger.info(
            "[LIVE] place_ifdoco order=%s symbol=%s side=%s qty=%.0f entry=%.3f sl=%.3f tp=%.3f meta=%s",
            order_id,
            symbol,
            side,
            qty,
            entry_px,
            sl_px,
            tp_px,
            json.dumps(meta, ensure_ascii=False, sort_keys=True),
            extra=_log_extra(opened_at),
        )
        return order_id

    def process_tick(self, row: Dict[str, Any], timestamp: float) -> List[Fill]:
        if self.dry_run and self._paper_delegate:
            return self._paper_delegate.process_tick(row, timestamp)
        return []

    def force_exit_order(
        self,
        order_id: str,
        timestamp: float,
        reason: str,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> Optional[Fill]:
        if self.dry_run and self._paper_delegate:
            return self._paper_delegate.force_exit_order(
                order_id, timestamp, reason, bid, ask
            )
        logger.warning(
            "[LIVE] force_exit_order stub order_id=%s reason=%s",
            order_id,
            reason,
            extra=_log_extra(timestamp),
        )
        return None

    def force_exit_all(self, reason: str, timestamp: float) -> List[Fill]:
        if self.dry_run and self._paper_delegate:
            return self._paper_delegate.force_exit_all(reason, timestamp)
        # TODO: Implement IFDOCO unwind via cancel/replace and status polling when live API wiring is added.
        logger.warning(
            "[LIVE] force_exit_all stub reason=%s",
            reason,
            extra=_log_extra(timestamp),
        )
        return []

    def list_open_orders(self) -> List[PaperOrder]:
        if self.dry_run and self._paper_delegate:
            return self._paper_delegate.list_open_orders()
        return []


class TradeLogger:
    ORDERS_LOG_CREATE = """
CREATE TABLE IF NOT EXISTS orders_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  symbol TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'runner',
  side TEXT NOT NULL,
  action TEXT NOT NULL,
  qty REAL NOT NULL,
  px REAL NOT NULL,
  reason TEXT NOT NULL,
  thr_md5 TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  mode TEXT NOT NULL,
  meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_log_symbol_ts ON orders_log(symbol, ts);
"""

    ORDERS_LOG_COLUMNS = {
        "px": "REAL",
        "type": "TEXT",
        "meta": "TEXT",
        "thr_md5": "TEXT",
        "dataset_id": "TEXT",
        "schema_version": "TEXT",
        "mode": "TEXT",
    }

    PAPER_PAIRS_CREATE = """
CREATE TABLE IF NOT EXISTS paper_pairs(
  order_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  qty REAL NOT NULL,
  entry_px REAL NOT NULL,
  sl_px REAL NOT NULL,
  tp_px REAL NOT NULL,
  exit_px REAL,
  exit_reason TEXT,
  opened_at REAL NOT NULL,
  closed_at REAL,
  fees REAL DEFAULT 0,
  realized_pnl REAL DEFAULT 0,
  max_unrealized_pnl REAL DEFAULT 0,
  min_unrealized_pnl REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pairs_symbol_time ON paper_pairs(symbol, opened_at);
"""

    PAPER_PAIRS_COLUMNS = {
        "order_id": "TEXT",
        "symbol": "TEXT",
        "side": "TEXT",
        "qty": "REAL",
        "entry_px": "REAL",
        "sl_px": "REAL",
        "tp_px": "REAL",
        "opened_at": "REAL DEFAULT 0",
        "exit_px": "REAL",
        "exit_reason": "TEXT",
        "closed_at": "REAL",
        "fees": "REAL DEFAULT 0",
        "realized_pnl": "REAL DEFAULT 0",
        "max_unrealized_pnl": "REAL DEFAULT 0",
        "min_unrealized_pnl": "REAL DEFAULT 0",
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()
        atexit.register(self.close)

    def _ensure_schema(self) -> None:
        self._ensure_table(
            "orders_log", self.ORDERS_LOG_CREATE, self.ORDERS_LOG_COLUMNS
        )
        self._ensure_table(
            "paper_pairs", self.PAPER_PAIRS_CREATE, self.PAPER_PAIRS_COLUMNS
        )

    def _ensure_table(
        self, name: str, create_sql: str, columns: Dict[str, str]
    ) -> None:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        if cur.fetchone() is None:
            self.conn.executescript(create_sql)
            self.conn.commit()
            return
        existing = {
            row["name"] for row in self.conn.execute(f"PRAGMA table_info({name})")
        }
        if name == "orders_log" and "px" not in existing:
            if "price" in existing:
                try:
                    self.conn.execute(
                        "ALTER TABLE orders_log RENAME COLUMN price TO px"
                    )
                    existing.add("px")
                except sqlite3.OperationalError as exc:
                    logger.debug("orders_log price->px rename failed: %s", exc)
            if "px" not in existing:
                self.conn.execute("ALTER TABLE orders_log ADD COLUMN px REAL")
                existing.add("px")
        if name == "orders_log" and "reason" not in existing:
            if "meta" in existing:
                try:
                    self.conn.execute("ALTER TABLE orders_log ADD COLUMN reason TEXT")
                    existing.add("reason")
                except sqlite3.OperationalError as exc:
                    logger.debug("orders_log add reason column failed: %s", exc)
        if name == "orders_log" and "reason" in existing:
            columns = dict(columns)
            columns.pop("reason", None)
        if name == "orders_log" and "type" not in existing:
            self.conn.execute(
                "ALTER TABLE orders_log ADD COLUMN type TEXT DEFAULT 'runner'"
            )
            existing.add("type")
        if name == "paper_pairs" and "order_id" not in existing:
            self.conn.execute("ALTER TABLE paper_pairs ADD COLUMN order_id TEXT")
            existing.add("order_id")
            columns = dict(columns)
            columns.pop("order_id", None)
        if name == "paper_pairs" and "opened_at" not in existing:
            self.conn.execute(
                "ALTER TABLE paper_pairs ADD COLUMN opened_at REAL DEFAULT 0"
            )
            existing.add("opened_at")
            columns = dict(columns)
            columns.pop("opened_at", None)
        for column, ddl in columns.items():
            if column not in existing:
                self.conn.execute(f"ALTER TABLE {name} ADD COLUMN {column} {ddl}")
        self.conn.commit()

    def log_entry(
        self,
        order_id: str,
        ts: float,
        symbol: str,
        side: str,
        qty: float,
        px: float,
        reason: str,
        profile_meta: dict,
        order_meta: dict,
    ):
        meta = dict(order_meta or {})
        meta["order_id"] = order_id

        self.conn.execute(
            """
            INSERT INTO orders_log (
                ts, symbol, side, action, qty, px, reason,
                thr_md5, dataset_id, schema_version, mode, meta
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,  # ts
                symbol,  # symbol
                side,  # side
                "entry",  # action
                qty,  # qty
                px,  # px
                reason,  # reason
                profile_meta.get("thr_md5", ""),
                profile_meta.get("dataset_id", ""),
                profile_meta.get("schema_version", ""),
                profile_meta.get("mode", ""),
                json.dumps(meta, ensure_ascii=False, sort_keys=True),
            ),
        )

        # paper_pairs は渡された値で明示的にINSERT（変数未定義を解消）
        self.conn.execute(
            """
            INSERT INTO paper_pairs(
                order_id, symbol, side, qty, entry_px, sl_px, tp_px,
                opened_at, fees, realized_pnl, max_unrealized_pnl, min_unrealized_pnl
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(order_id) DO UPDATE SET
                symbol=excluded.symbol,
                side=excluded.side,
                qty=excluded.qty,
                entry_px=excluded.entry_px,
                sl_px=excluded.sl_px,
                tp_px=excluded.tp_px,
                opened_at=excluded.opened_at
            """,
            (
                order_id,
                symbol,
                side,
                qty,
                px,
                order_meta.get("sl_px", px),
                order_meta.get("tp_px", px),
                ts,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        self.conn.commit()

    def log_exit(self, order_id: str, summary, profile_meta: dict, extra_meta: dict):
        meta = dict(extra_meta or {})
        meta["order_id"] = order_id

        self.conn.execute(
            """
            INSERT INTO orders_log (
                ts, symbol, side, action, qty, px, reason,
                thr_md5, dataset_id, schema_version, mode, meta
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.exit_ts,
                summary.symbol,
                summary.side,
                "exit",
                summary.qty,
                summary.exit_px,
                summary.exit_reason,
                profile_meta.get("thr_md5", ""),
                profile_meta.get("dataset_id", ""),
                profile_meta.get("schema_version", ""),
                profile_meta.get("mode", ""),
                json.dumps(meta, ensure_ascii=False, sort_keys=True),
            ),
        )

        self.conn.execute(
            """
            UPDATE paper_pairs
            SET exit_px=?, exit_reason=?, closed_at=?, fees=?, realized_pnl=?, max_unrealized_pnl=?, min_unrealized_pnl=?
            WHERE order_id=?
            """,
            (
                summary.exit_px,
                summary.exit_reason,
                summary.exit_ts,
                summary.fees,
                summary.realized_pnl,
                summary.max_unrealized,
                summary.min_unrealized,
                order_id,
            ),
        )
        self.conn.commit()

    @staticmethod
    def _meta_json(
        profile_meta: Dict[str, str], extra_meta: Optional[Dict[str, Any]]
    ) -> str:
        payload: Dict[str, Any] = dict(profile_meta)
        if extra_meta:
            payload.update(extra_meta)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
