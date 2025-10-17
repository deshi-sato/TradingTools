#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
grid_search_thresholds_sell.py

SELL 側のペーパールールを BUY 側と同じ評価パイプラインで最適化する。
ラベル/リターンは「下落で勝ち」となるように符号を調整し、IFDOCO 風の
シミュレーションで利確/損切/失速/タイムアウトを判定する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

JST = timezone(timedelta(hours=9))

MAX_HOLD_DEFAULT = 20.0
PATH_STEP_MS = 500
SPREAD_SPIKE_TICKS = 2


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        val = float(value)
        if math.isnan(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


@dataclass
class PathPoint:
    ts_ms: int
    dt: float
    mid: float
    v_rate: Optional[float]
    f1_delta: Optional[float]
    spread_ticks: Optional[float]


@dataclass
class DataRow:
    symbol: str
    ts_ms: int
    ts_sec: float
    horizon_sec: int
    ret_bp: float
    label: int
    down_ratio: Optional[float]
    score: Optional[float]
    spread_ticks: Optional[float]
    vol_surge: Optional[float]
    v_spike: Optional[float]
    v_rate: Optional[float]
    entry_bid: Optional[float]
    entry_ask: Optional[float]
    entry_mid: Optional[float]
    entry_f1_delta: Optional[float]
    group: str
    path: List[PathPoint] = field(default_factory=list)


def result_rank_key_sell(item: Dict[str, object]) -> Tuple:
    eligible_score = 1 if item.get("eligible") else 0
    params = item.get("params", {})
    return (
        eligible_score,
        float(item.get("precision") or 0.0),
        int(item.get("signals") or 0),
        float(item.get("ev") or 0.0),
        float(params.get("SELL_UPTICK_THR") or -math.inf),
        float(params.get("SELL_SCORE_THR") or -math.inf),
        float(params.get("VOLUME_SPIKE_THR") or -math.inf),
        -float(params.get("SELL_SPREAD_MAX") or math.inf),
        float(params.get("TP_BP") or -math.inf),
        -float(params.get("SL_BP") or math.inf),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for SELL-side thresholds.")
    parser.add_argument("-DatasetId", required=True)
    parser.add_argument("-Horizons", default="60,120")
    parser.add_argument("-MinTrades", type=int, default=50)
    parser.add_argument("-EVFloor", type=float, default=0.0)
    parser.add_argument("-CV", type=int, default=0)
    parser.add_argument("-Out")
    parser.add_argument("-Verbose", type=int, default=0)
    return parser.parse_args()


def parse_list(arg: str) -> List[int]:
    items = [p.strip() for p in arg.split(",") if p.strip()]
    if not items:
        raise SystemExit("ERROR: Horizons must not be empty")
    horizons: List[int] = []
    for item in items:
        try:
            val = int(item)
        except ValueError as exc:
            raise SystemExit(f"ERROR: invalid horizon '{item}'") from exc
        if val <= 0:
            raise SystemExit(f"ERROR: horizon must be positive (got {val})")
        horizons.append(val)
    return sorted(set(horizons))


def resolve_db_path(dataset_id: str) -> Path:
    db_dir = Path("db")
    candidates = sorted(db_dir.glob("naut_market_*_refeed.db"))
    if not candidates:
        raise SystemExit("ERROR: no refeed DBs found under db/")
    for path in candidates:
        try:
            conn = sqlite3.connect(str(path))
            cur = conn.execute(
                "SELECT source_db_path FROM dataset_registry WHERE dataset_id=?",
                (dataset_id,),
            )
            row = cur.fetchone()
            conn.close()
        except sqlite3.DatabaseError:
            continue
        if not row:
            continue
        source = row[0]
        final_path = Path(source) if source else path
        if not final_path.exists():
            final_path = path
        return final_path.resolve()
    raise SystemExit(f"ERROR: dataset_id {dataset_id} not found in registry.")


def feature_columns(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("PRAGMA table_info(features_stream)")
    return [row[1] for row in cur.fetchall()]


def ensure_indexes(conn: sqlite3.Connection, has_ts_ms: bool) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_labels_dataset_symbol_ts ON labels_outcome(dataset_id, symbol, ts)",
        "CREATE INDEX IF NOT EXISTS idx_labels_dataset_horizon ON labels_outcome(dataset_id, horizon_sec)",
    ]
    for stmt in statements:
        conn.execute(stmt)
    if has_ts_ms:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_symbol_ts_ms ON features_stream(symbol, ts_ms)"
        )
    else:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_symbol_ts ON features_stream(symbol, t_exec)"
        )
    conn.commit()


def resolve_effective_dataset_id(
    conn: sqlite3.Connection, requested_id: str
) -> Tuple[str, Optional[str]]:
    hit = conn.execute(
        "SELECT 1 FROM features_stream WHERE dataset_id=? LIMIT 1", (requested_id,)
    ).fetchone()
    if hit:
        return requested_id, None
    row = conn.execute(
        "SELECT dataset_id, COUNT(*) FROM features_stream GROUP BY dataset_id ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    if row and row[0]:
        return str(row[0]), requested_id
    return requested_id, None


def load_rows(
    conn: sqlite3.Connection,
    dataset_id: str,
    horizons: Sequence[int],
    has_vol_surge: bool,
    use_ts_ms: bool,
    has_v_spike: bool,
    has_v_rate: bool,
) -> List[DataRow]:
    if not horizons:
        return []
    placeholders = ",".join(["?"] * len(horizons))
    select_parts = [
        "l.symbol AS symbol",
        "l.ts AS ts_ms",
        "l.horizon_sec AS horizon_sec",
        "l.ret_bp AS ret_bp",
        "l.label AS label",
        "f.f1 AS f1",
        "f.score AS score",
        "f.spread_ticks AS spread_ticks",
        "f.bid1 AS bid1",
        "f.ask1 AS ask1",
        "f.f1_delta AS f1_delta",
        "f.v_rate AS v_rate",
    ]
    if has_vol_surge:
        select_parts.append("f.vol_surge_z AS vol_surge_z")
    if has_v_spike:
        select_parts.append("f.v_spike AS v_spike")
    join_condition = "f.symbol = l.symbol"
    if use_ts_ms:
        select_parts.append("f.ts_ms AS ts_ms_feat")
        join_condition += " AND f.ts_ms = l.ts"
    else:
        join_condition += " AND CAST(f.t_exec * 1000 AS INTEGER) = l.ts"
    sql = f"""
        SELECT {', '.join(select_parts)}
          FROM labels_outcome AS l
          JOIN features_stream AS f
            ON {join_condition}
         WHERE l.dataset_id = ?
           AND l.horizon_sec IN ({placeholders})
    """
    params: List[Any] = [dataset_id] + list(horizons)
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return []

    df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce")
    df["ret_bp"] = pd.to_numeric(df["ret_bp"], errors="coerce")
    df["horizon_sec"] = pd.to_numeric(df["horizon_sec"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["ts_ms", "ret_bp", "horizon_sec", "label"])
    df["ts_ms"] = df["ts_ms"].astype("int64")
    df["ts_sec"] = df["ts_ms"].astype("float64") / 1000.0
    df["horizon_sec"] = df["horizon_sec"].astype("int64")
    df["label"] = df["label"].astype("int64")

    numeric_optional = ["f1", "score", "spread_ticks", "bid1", "ask1", "f1_delta", "v_rate"]
    if has_vol_surge:
        numeric_optional.append("vol_surge_z")
    if has_v_spike:
        numeric_optional.append("v_spike")
    for col in numeric_optional:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # SELL 正規化: 下落で正、勝ちを label=1 に
    df["ret_bp"] = -df["ret_bp"]
    df["label"] = 1 - df["label"]

    ts_jst = pd.to_datetime(df["ts_sec"], unit="s") + pd.to_timedelta(9, unit="h")
    df["group"] = ts_jst.dt.strftime("%Y%m%d")
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "ts_ms", "horizon_sec"]).reset_index(drop=True)

    def _opt(val: Any) -> Optional[float]:
        return None if pd.isna(val) else float(val)

    rows: List[DataRow] = []
    for row in df.itertuples(index=False, name="Row"):
        entry_bid = _opt(getattr(row, "bid1", None))
        entry_ask = _opt(getattr(row, "ask1", None))
        entry_mid: Optional[float] = None
        if (
            entry_bid is not None
            and entry_ask is not None
            and entry_bid > 0
            and entry_ask > 0
        ):
            entry_mid = (entry_bid + entry_ask) * 0.5
        entry_f1_delta = _opt(getattr(row, "f1_delta", None))
        up_val = _opt(getattr(row, "f1", None))
        down_ratio = None if up_val is None else max(0.0, min(1.0, (10.0 - up_val) / 10.0))
        rows.append(
            DataRow(
                symbol=row.symbol,
                ts_ms=int(row.ts_ms),
                ts_sec=float(row.ts_sec),
                horizon_sec=int(row.horizon_sec),
                ret_bp=float(row.ret_bp),
                label=int(row.label),
                down_ratio=down_ratio,
                score=_opt(getattr(row, "score", None)),
                spread_ticks=_opt(getattr(row, "spread_ticks", None)),
                vol_surge=_opt(getattr(row, "vol_surge_z", None)),
                v_spike=_opt(getattr(row, "v_spike", None)) if has_v_spike else None,
                v_rate=_opt(getattr(row, "v_rate", None)) if has_v_rate else None,
                entry_bid=entry_bid,
                entry_ask=entry_ask,
                entry_mid=entry_mid,
                entry_f1_delta=entry_f1_delta,
                group=row.group,
            )
        )
    return rows


def attach_paths(
    conn: sqlite3.Connection,
    dataset_id: str,
    rows: Sequence[DataRow],
    max_hold_sec: float = MAX_HOLD_DEFAULT,
    step_ms: int = PATH_STEP_MS,
) -> None:
    if not rows:
        return
    symbols = sorted({row.symbol for row in rows})
    if not symbols:
        return
    max_hold_ms = int(max_hold_sec * 1000)
    min_ts = min(row.ts_ms for row in rows) - 1000
    max_ts = max(row.ts_ms for row in rows) + max_hold_ms + 1000
    symbol_placeholders = ",".join(["?"] * len(symbols))
    sql = f"""
        SELECT symbol, ts_ms, bid1, ask1, spread_ticks, v_rate, f1_delta
          FROM features_stream
         WHERE dataset_id = ?
           AND symbol IN ({symbol_placeholders})
           AND ts_ms BETWEEN ? AND ?
         ORDER BY symbol, ts_ms
    """
    params: List[Any] = [dataset_id] + symbols + [min_ts, max_ts]
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return
    for col in ("bid1", "ask1", "spread_ticks", "v_rate", "f1_delta"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["mid"] = (df["bid1"] + df["ask1"]) * 0.5
    grouped = {sym: frame for sym, frame in df.groupby("symbol")}

    for row in rows:
        frame = grouped.get(row.symbol)
        path: List[PathPoint] = []
        entry_mid = row.entry_mid
        path.append(
            PathPoint(
                ts_ms=row.ts_ms,
                dt=0.0,
                mid=entry_mid if entry_mid else 0.0,
                v_rate=row.v_rate,
                f1_delta=row.entry_f1_delta,
                spread_ticks=row.spread_ticks,
            )
        )
        if frame is None or entry_mid is None or entry_mid <= 0:
            row.path = path
            continue
        start_ts = row.ts_ms
        end_ts = start_ts + max_hold_ms
        last_ts = start_ts
        last_mid = entry_mid
        last_rate = row.v_rate
        last_f1 = row.entry_f1_delta
        last_spread = row.spread_ticks
        for _, rec in frame.iterrows():
            ts_ms = int(rec["ts_ms"])
            if ts_ms < start_ts or ts_ms > end_ts:
                continue
            if ts_ms - last_ts < step_ms and ts_ms != start_ts:
                continue
            mid_val = _safe_float(rec.get("mid"))
            if mid_val is None or mid_val <= 0:
                mid_val = last_mid
            if mid_val is None or mid_val <= 0:
                continue
            v_rate_val = _safe_float(rec.get("v_rate"))
            if v_rate_val is None:
                v_rate_val = last_rate
            f1_delta_val = _safe_float(rec.get("f1_delta"))
            if f1_delta_val is None:
                f1_delta_val = last_f1
            spread_val = _safe_float(rec.get("spread_ticks"))
            if spread_val is None:
                spread_val = last_spread
            dt = (ts_ms - start_ts) / 1000.0
            path.append(
                PathPoint(
                    ts_ms=ts_ms,
                    dt=dt,
                    mid=mid_val,
                    v_rate=v_rate_val,
                    f1_delta=f1_delta_val,
                    spread_ticks=spread_val,
                )
            )
            last_ts = ts_ms
            last_mid = mid_val
            last_rate = v_rate_val
            last_f1 = f1_delta_val
            last_spread = spread_val
            if dt >= max_hold_sec:
                break
        row.path = path


def group_by_day(rows: Sequence[DataRow]) -> Tuple[List[str], Dict[str, List[int]]]:
    day_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        day_groups[row.group].append(idx)
    days = sorted(day_groups.keys())
    return days, day_groups


def make_day_folds(days: Sequence[str], day_groups: Dict[str, List[int]], cv: int) -> List[List[str]]:
    if cv <= 1 or len(days) <= 1:
        return [list(days)]
    cv = max(2, min(cv, len(days)))
    folds: List[List[str]] = [[] for _ in range(cv)]
    fold_sizes = [0] * cv
    for day in days:
        size = len(day_groups.get(day, []))
        target = min(range(cv), key=lambda idx: fold_sizes[idx])
        folds[target].append(day)
        fold_sizes[target] += size
    return [fold for fold in folds if fold]


def _compute_sell_ret(entry_mid: float, current_mid: float) -> float:
    return (entry_mid - current_mid) / entry_mid * 10000.0


def simulate_trade_sell(row: DataRow, params: Dict[str, float]) -> Tuple[float, str]:
    entry_mid = row.entry_mid
    if entry_mid is None or entry_mid <= 0:
        return row.ret_bp, "fallback"
    path = row.path
    if not path or len(path) <= 1:
        return row.ret_bp, "no_path"

    tp_bp = float(params.get("TP_BP", 15.0))
    sl_bp = float(params.get("SL_BP", 8.0))
    stall_vrate_thr = params.get("STALL_VRATE_THR")
    stall_f1_thr = float(params.get("STALL_F1_THR", 0.0))
    stall_n = int(params.get("STALL_N", 2))
    max_hold_sec = float(params.get("MAX_HOLD_SEC", MAX_HOLD_DEFAULT))

    consecutive_stall = 0
    prev_spread = path[0].spread_ticks

    for point in path[1:]:
        current_mid = point.mid if point.mid > 0 else entry_mid
        ret_bp = _compute_sell_ret(entry_mid, current_mid)
        if ret_bp >= tp_bp:
            return ret_bp, "tp"
        if ret_bp <= -sl_bp:
            return ret_bp, "sl"

        spread = point.spread_ticks if point.spread_ticks is not None else prev_spread
        if (
            spread is not None
            and prev_spread is not None
            and (spread - prev_spread) >= SPREAD_SPIKE_TICKS
        ):
            return ret_bp, "spread"
        prev_spread = spread

        if stall_vrate_thr is not None:
            v_rate = point.v_rate if point.v_rate is not None else row.v_rate
            f1_delta = (
                point.f1_delta
                if point.f1_delta is not None
                else (row.entry_f1_delta if row.entry_f1_delta is not None else 0.0)
            )
            if v_rate is not None and v_rate < stall_vrate_thr and f1_delta >= stall_f1_thr:
                consecutive_stall += 1
                if consecutive_stall >= stall_n:
                    return ret_bp, "stall"
            else:
                consecutive_stall = 0

        if point.dt >= max_hold_sec:
            return ret_bp, "timeout"

    last_point = path[-1]
    last_mid = last_point.mid if last_point.mid > 0 else entry_mid
    ret_bp = _compute_sell_ret(entry_mid, last_mid)
    return ret_bp, "timeout"


def _zero_metrics() -> Dict[str, Any]:
    return {
        "signals": 0.0,
        "hits": 0.0,
        "sum_hit": 0.0,
        "sum_miss": 0.0,
        "loss_count": 0.0,
        "sum_ret": 0.0,
        "exit_counts": {},
    }


def evaluate_on_indices(
    rows: Sequence[DataRow],
    indices: Sequence[int],
    params: Dict[str, float],
    cooldown_sec: float,
) -> Dict[str, Any]:
    last_fire: Dict[str, float] = {}
    metrics = _zero_metrics()
    sorted_idx = sorted(
        indices, key=lambda i: (rows[i].symbol, rows[i].ts_ms, rows[i].horizon_sec)
    )

    down_thr = params.get("SELL_UPTICK_THR")
    score_thr = params.get("SELL_SCORE_THR")
    spread_max = params.get("SELL_SPREAD_MAX")
    vol_thr = params.get("VOL_SURGE_MIN")
    v_rate_thr = params.get("VOLUME_SPIKE_THR")

    for idx in sorted_idx:
        row = rows[idx]
        last = last_fire.get(row.symbol, -1e18)
        if row.ts_sec - last < cooldown_sec:
            continue

        if down_thr is not None and row.down_ratio is not None:
            if row.down_ratio < down_thr:
                continue
        if score_thr is not None and row.score is not None:
            if row.score < score_thr:
                continue
        if spread_max is not None and row.spread_ticks is not None:
            if row.spread_ticks > spread_max:
                continue
        if vol_thr is not None and row.vol_surge is not None:
            if row.vol_surge < vol_thr:
                continue
        if v_rate_thr is not None and row.v_rate is not None:
            if row.v_rate < v_rate_thr:
                continue

        last_fire[row.symbol] = row.ts_sec
        metrics["signals"] += 1.0
        trade_ret, exit_reason = simulate_trade_sell(row, params)
        metrics["exit_counts"].setdefault(exit_reason, 0.0)
        metrics["exit_counts"][exit_reason] += 1.0
        metrics["sum_ret"] += trade_ret
        if trade_ret > 0:
            metrics["hits"] += 1.0
            metrics["sum_hit"] += trade_ret
        elif trade_ret < 0:
            metrics["sum_miss"] += abs(trade_ret)
            metrics["loss_count"] += 1.0

    return metrics


def aggregate_metrics(fold_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_signals = sum(m.get("signals", 0.0) for m in fold_metrics)
    total_hits = sum(m.get("hits", 0.0) for m in fold_metrics)
    sum_hit = sum(m.get("sum_hit", 0.0) for m in fold_metrics)
    sum_miss = sum(m.get("sum_miss", 0.0) for m in fold_metrics)
    loss_count = sum(m.get("loss_count", 0.0) for m in fold_metrics)
    sum_ret = sum(m.get("sum_ret", 0.0) for m in fold_metrics)
    exit_counts: Dict[str, float] = defaultdict(float)
    for metric in fold_metrics:
        for key, val in metric.get("exit_counts", {}).items():
            exit_counts[key] += float(val)

    precision = total_hits / total_signals if total_signals > 0 else 0.0
    mean_hit = sum_hit / total_hits if total_hits > 0 else 0.0
    mean_miss = sum_miss / loss_count if loss_count > 0 else 0.0
    ev = sum_ret / total_signals if total_signals > 0 else float("-inf")
    return {
        "signals": total_signals,
        "hits": total_hits,
        "precision": precision,
        "mean_hit": mean_hit,
        "mean_miss": mean_miss,
        "ev": ev,
        "loss_count": loss_count,
        "sum_ret": sum_ret,
        "exit_counts": dict(exit_counts),
    }


def evaluate_params(
    rows: Sequence[DataRow],
    days: Sequence[str],
    day_groups: Dict[str, List[int]],
    params: Dict[str, float],
    cooldown_sec: float,
    cv: int,
) -> Dict[str, Any]:
    if cv <= 1 or len(days) <= 1:
        indices: List[int] = []
        for day in days:
            indices.extend(day_groups.get(day, []))
        if not indices:
            return aggregate_metrics([_zero_metrics()])
        return aggregate_metrics(
            [evaluate_on_indices(rows, indices, params, cooldown_sec)]
        )

    folds = make_day_folds(days, day_groups, cv)
    fold_metrics: List[Dict[str, Any]] = []
    for fold in folds:
        idxs: List[int] = []
        for day in fold:
            idxs.extend(day_groups.get(day, []))
        if not idxs:
            continue
        fold_metrics.append(evaluate_on_indices(rows, idxs, params, cooldown_sec))
    if not fold_metrics:
        fold_metrics.append(_zero_metrics())
    return aggregate_metrics(fold_metrics)


def grid_parameters(has_score: bool, has_vol: bool, has_v_rate: bool) -> List[Dict[str, float]]:
    upticks = [0.5, 0.55, 0.6]
    spreads = [15, 20, 25, 30]
    scores = [5.5, 6.0, 6.5, 7.0] if has_score else [None]
    cooldowns = [5.0, 10.0, 15.0]
    vol_surges = [0.0, 0.5, 1.0] if has_vol else [None]
    v_spikes = [1.6, 1.8, 2.0] if has_v_rate else [None]
    tp_candidates = [12.0, 15.0, 18.0]
    sl_candidates = [6.0, 8.0, 10.0]
    stall_vrate = [1.2, 1.4, 1.6] if has_v_rate else [None]
    stall_f1 = [0.0, 0.1]
    stall_n = [2]

    params_list: List[Dict[str, float]] = []
    for up in upticks:
        for sp in spreads:
            for sc in scores:
                for cd in cooldowns:
                    for vs in vol_surges:
                        for spike in v_spikes:
                            for tp in tp_candidates:
                                for sl in sl_candidates:
                                    for sv in stall_vrate:
                                        for sf in stall_f1:
                                            for sn in stall_n:
                                                params: Dict[str, float] = {
                                                    "SELL_UPTICK_THR": float(up),
                                                    "SELL_SPREAD_MAX": float(sp),
                                                    "COOLDOWN_SEC": float(cd),
                                                    "TP_BP": float(tp),
                                                    "SL_BP": float(sl),
                                                    "STALL_F1_THR": float(sf),
                                                    "STALL_N": float(sn),
                                                    "MAX_HOLD_SEC": MAX_HOLD_DEFAULT,
                                                }
                                                if sc is not None:
                                                    params["SELL_SCORE_THR"] = float(sc)
                                                if vs is not None:
                                                    params["VOL_SURGE_MIN"] = float(vs)
                                                if spike is not None:
                                                    params["VOLUME_SPIKE_THR"] = float(spike)
                                                if sv is not None:
                                                    params["STALL_VRATE_THR"] = float(sv)
                                                params_list.append(params)
    return params_list


def select_top(results: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
    return sorted(results, key=result_rank_key_sell, reverse=True)[:top_k]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_csv(path: Path, results: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "precision",
        "trades",
        "hits",
        "ev",
        "mean_hit_bp",
        "mean_loss_bp",
        "eligible",
        "sum_ret_bp",
        "SELL_UPTICK_THR",
        "SELL_SCORE_THR",
        "SELL_SPREAD_MAX",
        "COOLDOWN_SEC",
        "VOL_SURGE_MIN",
        "VOLUME_SPIKE_THR",
        "TP_BP",
        "SL_BP",
        "STALL_VRATE_THR",
        "STALL_F1_THR",
        "STALL_N",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\r\n")
        writer.writerow(headers)
        for result in results:
            params = result.get("params", {})
            writer.writerow(
                [
                    f"{result.get('precision', 0.0):.6f}",
                    int(result.get("signals", 0)),
                    int(result.get("hits", 0)),
                    f"{result.get('ev', 0.0):.6f}",
                    f"{result.get('mean_hit', 0.0):.6f}",
                    f"{result.get('mean_miss', 0.0):.6f}",
                    int(bool(result.get("eligible"))),
                    f"{result.get('sum_ret', 0.0):.6f}",
                    params.get("SELL_UPTICK_THR", ""),
                    params.get("SELL_SCORE_THR", ""),
                    params.get("SELL_SPREAD_MAX", ""),
                    params.get("COOLDOWN_SEC", ""),
                    params.get("VOL_SURGE_MIN", ""),
                    params.get("VOLUME_SPIKE_THR", ""),
                    params.get("TP_BP", ""),
                    params.get("SL_BP", ""),
                    params.get("STALL_VRATE_THR", ""),
                    params.get("STALL_F1_THR", ""),
                    params.get("STALL_N", ""),
                ]
            )


def compute_v_rate_summary(rows: Sequence[DataRow]) -> Dict[str, float]:
    values = [
        float(r.v_rate)
        for r in rows
        if r.v_rate is not None and not math.isnan(float(r.v_rate))
    ]
    if not values:
        return {}
    arr = np.asarray(values)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def evaluate_grid(
    rows: Sequence[DataRow],
    days: Sequence[str],
    day_groups: Dict[str, List[int]],
    params_list: Sequence[Dict[str, float]],
    min_trades: int,
    ev_floor: float,
    cv: int,
    verbose: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    best_so_far: Optional[Dict[str, object]] = None
    best_eligible: Optional[Dict[str, object]] = None

    total = len(params_list)
    print(f"[grid-sell] total candidates = {total}")
    sys.stdout.flush()

    start = time.time()
    for idx, params in enumerate(params_list, 1):
        cooldown = params.get("COOLDOWN_SEC", 5.0)
        metrics = evaluate_params(rows, days, day_groups, params, cooldown, cv)
        signals = int(metrics["signals"])
        precision = float(metrics["precision"])
        ev = float(metrics["ev"])
        eligible = signals >= min_trades and ev >= ev_floor
        result = {
            "params": params,
            "signals": signals,
            "hits": int(metrics["hits"]),
            "precision": precision,
            "mean_hit": metrics["mean_hit"],
            "mean_miss": metrics["mean_miss"],
            "ev": ev,
            "eligible": eligible,
            "sum_ret": metrics["sum_ret"],
            "exit_counts": metrics["exit_counts"],
        }
        results.append(result)
        if best_so_far is None or result_rank_key_sell(result) > result_rank_key_sell(best_so_far):
            best_so_far = result
        if eligible:
            if best_eligible is None or result_rank_key_sell(result) > result_rank_key_sell(best_eligible):
                best_eligible = result
        if verbose and idx % 500 == 0:
            elapsed = time.time() - start
            print(
                f"[grid-sell] {idx}/{total} ({idx/total*100:.1f}%) elapsed={elapsed:.1f}s "
                f"precision={precision:.4f} trades={signals} ev={ev:.3f} params={params}"
            )
    return results


def main() -> None:
    args = parse_args()
    horizons = parse_list(args.Horizons)
    db_path = resolve_db_path(args.DatasetId)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    columns = set(feature_columns(conn))
    has_ts_ms = "ts_ms" in columns
    ensure_indexes(conn, has_ts_ms)

    effective_id, override = resolve_effective_dataset_id(conn, args.DatasetId)
    if override:
        print(f"[grid-sell] dataset_id override: requested {override} -> {effective_id}")

    has_vol_surge = "vol_surge_z" in columns
    has_v_spike = "v_spike" in columns
    has_v_rate = "v_rate" in columns

    rows = load_rows(
        conn,
        effective_id,
        horizons,
        has_vol_surge=has_vol_surge,
        use_ts_ms=has_ts_ms,
        has_v_spike=has_v_spike,
        has_v_rate=has_v_rate,
    )
    attach_paths(conn, effective_id, rows)
    conn.close()

    if not rows:
        raise SystemExit("ERROR: no rows joined between features_stream and labels_outcome.")

    days, day_groups = group_by_day(rows)
    params_list = grid_parameters(
        has_score="score" in columns,
        has_vol=has_vol_surge,
        has_v_rate=has_v_rate,
    )

    v_rate_summary = compute_v_rate_summary(rows) if has_v_rate else {}
    if v_rate_summary:
        print(
            "[grid-sell] v_rate stats: "
            f"p50={v_rate_summary['p50']:.2f} "
            f"p95={v_rate_summary['p95']:.2f} "
            f"p99={v_rate_summary['p99']:.2f} "
            f"max={v_rate_summary['max']:.2f} "
            f"(n={v_rate_summary['count']})"
        )

    results = evaluate_grid(
        rows,
        days,
        day_groups,
        params_list,
        min_trades=max(1, int(args.MinTrades)),
        ev_floor=float(args.EVFloor),
        cv=max(0, int(args.CV)),
        verbose=bool(args.Verbose),
    )

    ordered = sorted(results, key=result_rank_key_sell, reverse=True)
    top = select_top(ordered, top_k=5)
    for i, item in enumerate(top, 1):
        print(
            f"[grid-top-SELL] #{i} precision={item['precision']:.4f} trades={item['signals']} "
            f"EV={item['ev']:.3f} params={item['params']}"
        )

    best = top[0] if top else (ordered[0] if ordered else {})
    dataset_id = effective_id
    out_json = Path(args.Out) if args.Out else Path(f"exports/best_thresholds_sell_{dataset_id}.json")
    write_json(out_json, best)
    print(f"[grid-sell] best saved to {out_json}")

    out_csv = Path(f"exports/grid_report_sell_{dataset_id}.csv")
    write_csv(out_csv, ordered)
    print(f"[grid-sell] report saved to {out_csv}")


if __name__ == "__main__":
    main()
