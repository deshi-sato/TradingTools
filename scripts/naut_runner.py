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
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.common_config import load_json_utf8

REPO_ROOT = Path(__file__).resolve().parents[1]
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"
ACTION_ENTRY = "entry"
ACTION_EXIT = "exit"

logger = logging.getLogger(__name__)

_singleton_handle: Optional[int] = None
_pidfile_path: Optional[Path] = None


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


def now_ts() -> float:
    return time.time()


@dataclass(frozen=True)
class RunnerConfig:
    features_db: str
    ops_db: str
    symbols: List[str]
    poll_interval_sec: float = 1.0
    initial_cash: float = 10_000_000.0
    fee_rate_bps: float = 0.0
    slippage_ticks: float = 0.0
    tick_size: float = 0.1
    tick_value: float = 1.0
    min_lot: int = 100
    risk_per_trade_pct: float = 0.01
    max_concurrent_positions: int = 1
    daily_loss_limit_pct: float = 0.05
    stats_interval_sec: float = 300.0
    stop_loss_ticks: float = 5.0
    log_path: str = "logs/naut_runner.log"
    timezone: str = "Asia/Tokyo"
    killswitch_check_interval_sec: float = 5.0


def load_runner_config(config_path: Path) -> RunnerConfig:
    payload = load_json_utf8(str(config_path))
    try:
        symbols_raw = payload["symbols"]
    except KeyError as exc:
        raise SystemExit(f"Runner config missing symbols: {config_path}") from exc
    if not isinstance(symbols_raw, list) or not symbols_raw:
        raise SystemExit("Runner config requires non-empty list of symbols")
    features_db = resolve_path(payload.get("features_db", "naut_market.db"))
    ops_db = resolve_path(payload.get("ops_db", "naut_ops.db"))
    log_path = payload.get("log_path", "logs/naut_runner.log")
    if not Path(log_path).is_absolute():
        log_path = str((REPO_ROOT / log_path).resolve())
    return RunnerConfig(
        features_db=features_db,
        ops_db=ops_db,
        symbols=[str(sym) for sym in symbols_raw],
        poll_interval_sec=float(payload.get("poll_interval_sec", 1.0)),
        initial_cash=float(payload.get("initial_cash", 10_000_000.0)),
        fee_rate_bps=float(payload.get("fee_rate_bps", 0.0)),
        slippage_ticks=float(payload.get("slippage_ticks", 0.0)),
        tick_size=float(payload.get("tick_size", 0.1)),
        tick_value=float(payload.get("tick_value", 1.0)),
        min_lot=int(payload.get("min_lot", 100)),
        risk_per_trade_pct=float(payload.get("risk_per_trade_pct", 0.01)),
        max_concurrent_positions=int(payload.get("max_concurrent_positions", 1)),
        daily_loss_limit_pct=float(payload.get("daily_loss_limit_pct", 0.05)),
        stats_interval_sec=float(payload.get("stats_interval_sec", 300.0)),
        stop_loss_ticks=float(payload.get("stop_loss_ticks", 5.0)),
        log_path=log_path,
        timezone=str(payload.get("timezone", "Asia/Tokyo")),
        killswitch_check_interval_sec=float(payload.get("killswitch_check_interval_sec", 5.0)),
    )


@dataclass(frozen=True)
class ThresholdProfile:
    dataset_id: str
    schema_version: str
    md5: str
    created_at: str
    mode: str
    uptick_thr: float
    score_thr: float
    spread_max: float
    volume_spike_thr: float
    cooldown_sec: float
    runner_max_hold_sec: float
    rr_tp_sl: float
    extras: Dict[str, Any] = field(default_factory=dict)

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
    raw = load_json_utf8(str(path))
    json_md5 = str(raw.get("md5", computed_md5))
    if json_md5 != computed_md5:
        logger.warning("Threshold md5 mismatch: file=%s payload=%s", computed_md5, json_md5)
    mode_raw = str(raw.get("mode", raw.get("MODE", SIDE_BUY))).upper()
    created_at = str(raw.get("created_at", datetime.utcnow().isoformat()))
    known = {
        "DATASET_ID",
        "SCHEMA_VERSION",
        "MD5",
        "CREATED_AT",
        "MODE",
        "UPTICK_THR",
        "SCORE_THR",
        "SPREAD_MAX",
        "VOLUME_SPIKE_THR",
        "COOLDOWN_SEC",
        "RUNNER_MAX_HOLD_SEC",
        "RR_TP_SL",
    }
    extras = {k: v for k, v in raw.items() if k.upper() not in known}
    profile = ThresholdProfile(
        dataset_id=str(raw.get("dataset_id", "")),
        schema_version=str(raw.get("schema_version", "")),
        md5=computed_md5,
        created_at=created_at,
        mode=mode_raw or SIDE_BUY,
        uptick_thr=float(raw.get("UPTICK_THR", raw.get("uptick_thr", 0.0))),
        score_thr=float(raw.get("SCORE_THR", raw.get("score_thr", 0.0))),
        spread_max=float(raw.get("SPREAD_MAX", raw.get("spread_max", 0.0))),
        volume_spike_thr=float(raw.get("VOLUME_SPIKE_THR", raw.get("volume_spike_thr", 0.0))),
        cooldown_sec=float(raw.get("COOLDOWN_SEC", raw.get("cooldown_sec", 0.0))),
        runner_max_hold_sec=float(raw.get("RUNNER_MAX_HOLD_SEC", raw.get("runner_max_hold_sec", 0.0))),
        rr_tp_sl=float(raw.get("RR_TP_SL", raw.get("rr_tp_sl", 2.0))) or 1.0,
        extras=extras,
    )
    return profile


@dataclass
class PolicyDecision:
    should_enter: bool
    entry_px: float = 0.0
    sl_px: float = 0.0
    tp_px: float = 0.0
    reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


class Policy:
    def __init__(self, profile: ThresholdProfile, mode: str, config: RunnerConfig):
        self.profile = profile
        self.mode = mode.upper()
        self.config = config
        if profile.mode.upper() != self.mode:
            logger.warning(
                "Threshold mode %s mismatches runner mode %s",
                profile.mode,
                self.mode,
            )

    @property
    def cooldown_sec(self) -> float:
        return max(self.profile.cooldown_sec, 0.0)

    @property
    def max_hold_sec(self) -> float:
        return max(self.profile.runner_max_hold_sec, 0.0)

    def evaluate(self, symbol: str, row: Dict[str, Any]) -> PolicyDecision:
        entry_price = self._entry_price(row)
        if entry_price is None:
            return PolicyDecision(False, reason="no_quote")
        score = self._extract(row, ("score",))
        uptick = self._extract(row, ("uptick_ratio", "uptick", "f1"))
        downtick = self._extract(row, ("downtick_ratio", "downtick", "f2"))
        spread = self._extract(row, ("spread_ticks", "spread", "spread_tick"))
        volume = self._extract(row, ("volume_rate", "v_rate", "volume_spike", "volume_ratio"))

        context = {
            "symbol": symbol,
            "score": score,
            "uptick_ratio": uptick,
            "downtick_ratio": downtick,
            "spread_ticks": spread,
            "volume_rate": volume,
            "entry_px": entry_price,
        }

        if spread is None:
            spread = float("inf")
        if volume is None:
            volume = 0.0

        if self.mode == SIDE_BUY:
            score_thr = abs(self.profile.score_thr)
            if score is None or score < score_thr:
                return PolicyDecision(False, reason="score", context=context)
            if uptick is None or uptick < self.profile.uptick_thr:
                return PolicyDecision(False, reason="uptick", context=context)
            if spread > self.profile.spread_max:
                return PolicyDecision(False, reason="spread", context=context)
            if volume < self.profile.volume_spike_thr:
                return PolicyDecision(False, reason="volume", context=context)
            sl_distance = self.config.stop_loss_ticks * self.config.tick_size
            if sl_distance <= 0:
                return PolicyDecision(False, reason="sl_distance", context=context)
            tp_distance = max(self.profile.rr_tp_sl * sl_distance, self.config.tick_size)
            sl_px = entry_price - sl_distance
            tp_px = entry_price + tp_distance
        else:
            raw_thr = self.profile.score_thr
            # SELL thresholds sometimes persist as positive magnitudes; normalise to the
            # expected negative domain so policy aligns with grid-search semantics.
            score_thr = raw_thr if raw_thr < 0 else -abs(raw_thr)
            if score is None or score >= score_thr:
                return PolicyDecision(False, reason="score", context=context)
            if downtick is None or downtick < self.profile.uptick_thr:
                return PolicyDecision(False, reason="downtick", context=context)
            if spread > self.profile.spread_max:
                return PolicyDecision(False, reason="spread", context=context)
            if volume < self.profile.volume_spike_thr:
                return PolicyDecision(False, reason="volume", context=context)
            sl_distance = self.config.stop_loss_ticks * self.config.tick_size
            if sl_distance <= 0:
                return PolicyDecision(False, reason="sl_distance", context=context)
            tp_distance = max(self.profile.rr_tp_sl * sl_distance, self.config.tick_size)
            sl_px = entry_price + sl_distance
            tp_px = entry_price - tp_distance

        context["sl_px"] = sl_px
        context["tp_px"] = tp_px
        return PolicyDecision(True, entry_price, sl_px, tp_px, reason="signal", context=context)

    @staticmethod
    def _extract(row: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
        for key in keys:
            if key in row:
                value = safe_float(row.get(key))
                if value is not None:
                    return value
        return None

    def _entry_price(self, row: Dict[str, Any]) -> Optional[float]:
        if self.mode == SIDE_BUY:
            for key in ("ask1", "ask", "ask_px", "ask_price"):
                price = safe_float(row.get(key))
                if price is not None:
                    return price
        else:
            for key in ("bid1", "bid", "bid_px", "bid_price"):
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

    def fetch_since(self, symbol: str, last_ts: float, limit: int = 500) -> List[Dict[str, Any]]:
        query = f"""
SELECT *
FROM {self.table}
WHERE symbol=? AND t_exec>?
ORDER BY t_exec ASC
LIMIT ?
"""
        cur = self.conn.execute(query, (symbol, last_ts, limit))
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


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
    ):
        self.mode = mode.upper()
        self.config = config
        self.policy = policy
        self.broker = broker
        self.ledger = ledger
        self.poller = poller
        self.trade_logger = trade_logger
        self.flatten_at = flatten_at
        self.flatten_triggered = False
        self.killswitch_path = killswitch_path
        self.last_killswitch_check = 0.0
        self.stop_requested = False
        self.cooldown_until: Dict[str, float] = {sym: 0.0 for sym in config.symbols}
        self.last_ts: Dict[str, float] = {sym: 0.0 for sym in config.symbols}
        self.stats: Dict[str, Any] = {
            "polled": 0,
            "signals": 0,
            "entries": 0,
            "exits": 0,
            "wins": 0,
            "losses": 0,
            "pnl_sum": 0.0,
        }
        self.last_stats_log = now_ts()
        self.profile = policy.profile
        self.loss_limit_engaged = False

    def run(self) -> None:
        logger.info("Runner loop start symbols=%s", ",".join(self.config.symbols))
        try:
            while not self.stop_requested:
                loop_start = now_ts()
                self._check_flatten()
                self._check_killswitch(loop_start)
                self._handle_loss_limit()
                if self.stop_requested:
                    break
                for symbol in self.config.symbols:
                    rows = self.poller.fetch_since(symbol, self.last_ts[symbol])
                    if not rows:
                        continue
                    for row in rows:
                        ts = safe_float(row.get("t_exec"), loop_start) or loop_start
                        self.last_ts[symbol] = max(self.last_ts[symbol], ts)
                        self.stats["polled"] += 1
                        fills = self.broker.process_tick(row, ts)
                        if fills:
                            self._handle_fills(fills)
                        self._mark_positions(symbol, row)
                        self._check_timeouts(symbol, ts, row)
                        self._try_entry(symbol, row, ts)
                self._maybe_log_stats(now_ts())
                sleep_for = max(0.0, self.config.poll_interval_sec - (now_ts() - loop_start))
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
            summary = self.ledger.register_exit(fill.order_id, fill.exit_px, fill.exit_reason, fill.timestamp)
            if not summary:
                continue
            self.stats["exits"] += 1
            self.stats["pnl_sum"] += summary.realized_pnl
            if summary.realized_pnl > 0:
                self.stats["wins"] += 1
            elif summary.realized_pnl < 0:
                self.stats["losses"] += 1
            self.cooldown_until[summary.symbol] = fill.timestamp + self.policy.cooldown_sec
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
            self.trade_logger.log_exit(fill.order_id, summary, self.profile.meta, extra_meta)
            logger.info(
                "EXIT %s order=%s side=%s qty=%.0f px=%.3f reason=%s pnl=%.2f md5=%s dataset=%s schema=%s",
                summary.symbol,
                summary.order_id,
                summary.side,
                summary.qty,
                summary.exit_px,
                summary.exit_reason,
                summary.realized_pnl,
                self.profile.md5,
                self.profile.dataset_id,
                self.profile.schema_version,
            )

    def _mark_positions(self, symbol: str, row: Dict[str, Any]) -> None:
        bid = safe_float(row.get("bid1"), safe_float(row.get("bid"), safe_float(row.get("bid_px"))))
        ask = safe_float(row.get("ask1"), safe_float(row.get("ask"), safe_float(row.get("ask_px"))))
        if bid is None and ask is None:
            price = safe_float(row.get("price"), safe_float(row.get("last_price")))
            bid = ask = price
        self.ledger.mark_symbol(symbol, bid, ask)

    def _check_timeouts(self, symbol: str, timestamp: float, row: Dict[str, Any]) -> None:
        if self.policy.max_hold_sec <= 0:
            return
        timeout_candidates = [
            position
            for position in list(self.ledger.positions.values())
            if position.symbol == symbol and (timestamp - position.opened_at) >= self.policy.max_hold_sec
        ]
        if not timeout_candidates:
            return
        bid = safe_float(row.get("bid1"), safe_float(row.get("bid"), safe_float(row.get("bid_px"))))
        ask = safe_float(row.get("ask1"), safe_float(row.get("ask"), safe_float(row.get("ask_px"))))
        fills: List[Fill] = []
        for position in timeout_candidates:
            fill = self.broker.force_exit_order(position.order_id, timestamp, "timeout", bid, ask)
            if fill:
                fills.append(fill)
        if fills:
            self._handle_fills(fills)

    def _try_entry(self, symbol: str, row: Dict[str, Any], timestamp: float) -> None:
        if not self.ledger.can_open(symbol):
            return
        if timestamp < self.cooldown_until.get(symbol, 0.0):
            return
        if self.ledger.loss_limit_hit:
            return
        decision = self.policy.evaluate(symbol, row)
        if not decision.should_enter:
            return
        self.stats["signals"] += 1
        qty = self.ledger.compute_order_size(decision.entry_px, decision.sl_px)
        if qty <= 0:
            logger.debug("Skip entry %s qty=%s (sizing filtered)", symbol, qty)
            return
        order_meta = {
            "thr_md5": self.profile.md5,
            "dataset_id": self.profile.dataset_id,
            "schema_version": self.profile.schema_version,
            "mode": self.profile.mode,
            "score": decision.context.get("score"),
            "uptick_ratio": decision.context.get("uptick_ratio"),
            "downtick_ratio": decision.context.get("downtick_ratio"),
            "spread_ticks": decision.context.get("spread_ticks"),
            "volume_rate": decision.context.get("volume_rate"),
            "entry_px": decision.entry_px,
            "sl_px": decision.sl_px,
            "tp_px": decision.tp_px,
        }
        order_id = self.broker.place_ifdoco(
            symbol,
            self.mode,
            qty,
            decision.entry_px,
            decision.sl_px,
            decision.tp_px,
            order_meta,
            timestamp,
        )
        self.ledger.register_entry(
            order_id,
            symbol,
            self.mode,
            qty,
            decision.entry_px,
            decision.sl_px,
            decision.tp_px,
            timestamp,
            order_meta,
        )
        self.trade_logger.log_entry(
            order_id,
            timestamp,
            symbol,
            self.mode,
            qty,
            decision.entry_px,
            decision.reason,
            self.profile.meta,
            order_meta,
        )
        self.stats["entries"] += 1
        logger.info(
            "ENTRY %s order=%s side=%s qty=%.0f entry=%.3f sl=%.3f tp=%.3f score=%s md5=%s dataset=%s schema=%s",
            symbol,
            order_id,
            self.mode,
            qty,
            decision.entry_px,
            decision.sl_px,
            decision.tp_px,
            decision.context.get("score"),
            self.profile.md5,
            self.profile.dataset_id,
            self.profile.schema_version,
        )

    def _check_flatten(self) -> None:
        if self.flatten_at is None or self.flatten_triggered:
            return
        current_time = datetime.now().time()
        if current_time >= self.flatten_at:
            logger.warning("Flatten triggered at %s", current_time.strftime("%H:%M"))
            self._flatten("flatten")
            self.flatten_triggered = True
            self.stop_requested = True

    def _check_killswitch(self, now: float) -> None:
        if now - self.last_killswitch_check < self.config.killswitch_check_interval_sec:
            return
        self.last_killswitch_check = now
        if self.killswitch_path.exists():
            logger.error("Killswitch detected: %s", self.killswitch_path)
            self._flatten("killswitch")
            self.stop_requested = True
            try:
                self.killswitch_path.unlink()
            except Exception:
                pass

    def _handle_loss_limit(self) -> None:
        if self.ledger.loss_limit_hit and not self.loss_limit_engaged:
            self.loss_limit_engaged = True
            drawdown_pct = 0.0
            if self.ledger.initial_cash > 0:
                drawdown_pct = (self.ledger.initial_cash - self.ledger.equity) / self.ledger.initial_cash * 100
            logger.error(
                "Daily loss limit hit: equity=%.2f initial=%.2f drawdown=%.2f%% -> flatten",
                self.ledger.equity,
                self.ledger.initial_cash,
                drawdown_pct,
            )
            self._flatten("loss_limit")
            self.stop_requested = True

    def _flatten(self, reason: str) -> None:
        fills = self.broker.force_exit_all(reason, now_ts())
        if fills:
            self._handle_fills(fills)
        else:
            logger.warning("Flatten requested (%s) but broker returned no fills", reason)

    def _maybe_log_stats(self, now: float) -> None:
        if now - self.last_stats_log < self.config.stats_interval_sec:
            return
        exits = self.stats["exits"]
        win_rate = (self.stats["wins"] / exits) if exits else 0.0
        ev_est = (self.stats["pnl_sum"] / exits) if exits else 0.0
        drawdown_pct = 0.0
        if self.ledger.peak_equity > 0:
            drawdown_pct = (self.ledger.drawdown / self.ledger.peak_equity) * 100.0
        logger.info(
            "stats: polled=%d signals=%d entries=%d exits=%d win_rate=%.2f ev_est=%.2f equity=%.2f drawdown=%.2f drawdown_pct=%.2f%%",
            self.stats["polled"],
            self.stats["signals"],
            self.stats["entries"],
            exits,
            win_rate,
            ev_est,
            self.ledger.equity,
            self.ledger.drawdown,
            drawdown_pct,
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
        self.last_stats_log = now


def configure_logging(log_path: str, verbose: bool) -> None:
    ensure_parent_dir(log_path)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Naut Runner IFDOCO executor")
    parser.add_argument("--mode", choices=[SIDE_BUY, SIDE_SELL], required=True)
    parser.add_argument("--thr", required=True, help="Path to best_thresholds_*_latest.json")
    parser.add_argument("--broker", choices=["paper", "live"], default="paper")
    parser.add_argument("--dry-run", type=int, choices=[0, 1], default=1, help="Live only: 0=live orders, 1=dry-run")
    parser.add_argument("--config", required=True, help="Runner config JSON")
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=0)
    parser.add_argument("--flatten-at", dest="flatten_at", help="HH:MM local time to flatten positions")
    parser.add_argument("--killswitch", default=str(REPO_ROOT / "runtime" / "stop.flag"))
    args = parser.parse_args()

    singleton_guard(f"naut_runner_{args.mode.lower()}_{args.broker}")

    config_path = Path(resolve_path(args.config))
    runner_config = load_runner_config(config_path)
    configure_logging(runner_config.log_path, bool(args.verbose))

    threshold_path = Path(resolve_path(args.thr))
    threshold_profile = load_threshold_profile(threshold_path)

    flatten_at = parse_flatten_at(args.flatten_at)
    killswitch_path = Path(resolve_path(args.killswitch))

    poller = FeaturePoller(runner_config.features_db)
    ledger = Ledger(runner_config)
    policy = Policy(threshold_profile, args.mode, runner_config)
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
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()


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
        risk_cash = equity * self.config.risk_per_trade_pct
        if risk_cash <= 0:
            return 0
        raw_qty = risk_cash / max(risk_per_unit, 1e-9)
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

    def mark_symbol(self, symbol: str, bid: Optional[float], ask: Optional[float]) -> None:
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
        diff = (mark_px - position.entry_px) if position.side == SIDE_BUY else (position.entry_px - mark_px)
        pnl = diff * position.qty * self.config.tick_value
        position.unrealized = pnl
        position.max_unrealized = max(position.max_unrealized, pnl)
        position.min_unrealized = min(position.min_unrealized, pnl)
        self._recalc_unrealized()
        self._update_equity()

    def register_exit(self, order_id: str, exit_px: float, reason: str, exit_ts: float) -> Optional[ExitSummary]:
        position = self.positions.pop(order_id, None)
        if not position:
            return None
        self.positions_by_symbol.pop(position.symbol, None)
        diff = (exit_px - position.entry_px) if position.side == SIDE_BUY else (position.entry_px - exit_px)
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
        self.unrealized_pnl = sum(position.unrealized for position in self.positions.values())

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
        )
        return order_id

    def process_tick(self, row: Dict[str, Any], timestamp: float) -> List[Fill]:
        symbol = row.get("symbol")
        if not symbol:
            return []
        bid = safe_float(row.get("bid1"), safe_float(row.get("bid"), safe_float(row.get("bid_px"))))
        ask = safe_float(row.get("ask1"), safe_float(row.get("ask"), safe_float(row.get("ask_px"))))
        fills: List[Fill] = []
        for order in list(self._orders.values()):
            if order.symbol != symbol:
                continue
            if bid is not None:
                order.last_bid = bid
            if ask is not None:
                order.last_ask = ask
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

    def _finalize(self, order: PaperOrder, exit_px: float, reason: str, timestamp: float) -> Fill:
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
            )
            return self._paper_delegate.place_ifdoco(symbol, side, qty, entry_px, sl_px, tp_px, meta, opened_at)
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
            return self._paper_delegate.force_exit_order(order_id, timestamp, reason, bid, ask)
        logger.warning("[LIVE] force_exit_order stub order_id=%s reason=%s", order_id, reason)
        return None

    def force_exit_all(self, reason: str, timestamp: float) -> List[Fill]:
        if self.dry_run and self._paper_delegate:
            return self._paper_delegate.force_exit_all(reason, timestamp)
        # TODO: Implement IFDOCO unwind via cancel/replace and status polling when live API wiring is added.
        logger.warning("[LIVE] force_exit_all stub reason=%s", reason)
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
        self._ensure_table("orders_log", self.ORDERS_LOG_CREATE, self.ORDERS_LOG_COLUMNS)
        self._ensure_table("paper_pairs", self.PAPER_PAIRS_CREATE, self.PAPER_PAIRS_COLUMNS)

    def _ensure_table(self, name: str, create_sql: str, columns: Dict[str, str]) -> None:
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
        if cur.fetchone() is None:
            self.conn.executescript(create_sql)
            self.conn.commit()
            return
        existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({name})")}
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
        profile_meta: Dict[str, str],
        extra_meta: Dict[str, Any],
    ) -> None:
        meta_json = self._meta_json(profile_meta, extra_meta)
        self.conn.execute(
            """
            INSERT INTO orders_log(
                ts, symbol, side, action, qty, px, reason,
                thr_md5, dataset_id, schema_version, mode, meta
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ts,
                symbol,
                side,
                ACTION_ENTRY,
                qty,
                px,
                reason,
                profile_meta["thr_md5"],
                profile_meta["dataset_id"],
                profile_meta["schema_version"],
                profile_meta["mode"],
                meta_json,
            ),
        )
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
                extra_meta.get("entry_px", px),
                extra_meta.get("sl_px", px),
                extra_meta.get("tp_px", px),
                ts,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        self.conn.commit()

    def log_exit(
        self,
        order_id: str,
        summary: ExitSummary,
        profile_meta: Dict[str, str],
        extra_meta: Dict[str, Any],
    ) -> None:
        meta_json = self._meta_json(profile_meta, extra_meta)
        self.conn.execute(
            """
            INSERT INTO orders_log(
                ts, symbol, side, action, qty, px, reason,
                thr_md5, dataset_id, schema_version, mode, meta
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                summary.exit_ts,
                summary.symbol,
                summary.side,
                ACTION_EXIT,
                summary.qty,
                summary.exit_px,
                summary.exit_reason,
                profile_meta["thr_md5"],
                profile_meta["dataset_id"],
                profile_meta["schema_version"],
                profile_meta["mode"],
                meta_json,
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
    def _meta_json(profile_meta: Dict[str, str], extra_meta: Optional[Dict[str, Any]]) -> str:
        payload: Dict[str, Any] = dict(profile_meta)
        if extra_meta:
            payload.update(extra_meta)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
