#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
grid_search_thresholds.py

Performs a grid search over paper-rule thresholds using features_stream + labels_outcome.
Evaluates precision via day-based GroupKFold and persists the best parameter set.
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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

JST = timezone(timedelta(hours=9))


@dataclass
class DataRow:
    symbol: str
    ts_ms: int
    ts_sec: float
    horizon_sec: int
    ret_bp: float
    label: int
    uptick: Optional[float]
    score: Optional[float]
    spread_bp: Optional[float]
    vol_surge: Optional[float]
    group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search paper-rule thresholds.")
    parser.add_argument("-DatasetId", required=True)
    parser.add_argument("-Horizons", default="60,120")
    parser.add_argument("-MinTrades", type=int, default=150)
    parser.add_argument("-EVFloor", type=float, default=0.0)
    parser.add_argument("-CV", type=int, default=0)
    parser.add_argument("-Out")
    parser.add_argument("-Verbose", type=int, default=0)
    return parser.parse_args()


def parse_list(arg: str) -> List[int]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise SystemExit("ERROR: Horizons must not be empty")
    result: List[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise SystemExit(f"ERROR: invalid horizon value '{part}'") from exc
        if value <= 0:
            raise SystemExit(f"ERROR: horizon must be positive (got {value})")
        result.append(value)
    return sorted(set(result))


def resolve_db_path(dataset_id: str) -> Path:
    db_dir = Path("db")
    candidates = sorted(db_dir.glob("naut_market_*_refeed.db"))
    if not candidates:
        raise SystemExit("ERROR: no refeed DBs under db/")
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


def ensure_indexes(conn: sqlite3.Connection) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_labels_dataset_symbol_ts ON labels_outcome(dataset_id, symbol, ts)",
        "CREATE INDEX IF NOT EXISTS idx_labels_dataset_horizon ON labels_outcome(dataset_id, horizon_sec)",
        "CREATE INDEX IF NOT EXISTS idx_features_symbol_ts ON features_stream(symbol, t_exec)",
    ]
    for stmt in statements:
        conn.execute(stmt)
    conn.commit()


def compute_spread_bp(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    diff = ask - bid
    if diff < 0:
        return None
    return (diff / bid) * 10000.0


def load_rows(
    conn: sqlite3.Connection,
    dataset_id: str,
    horizons: Sequence[int],
    has_vol_surge: bool,
) -> List[DataRow]:
    placeholders = ",".join(["?"] * len(horizons))
    select_parts = [
        "l.symbol",
        "l.ts",
        "l.horizon_sec",
        "l.ret_bp",
        "l.label",
        "f.t_exec",
        "f.f1",
        "f.score",
        "f.bid1",
        "f.ask1",
    ]
    if has_vol_surge:
        select_parts.append("f.vol_surge_z")
    sql = f"""
        SELECT {', '.join(select_parts)}
          FROM labels_outcome AS l
          JOIN features_stream AS f
            ON f.symbol = l.symbol
           AND CAST(f.t_exec * 1000 AS INTEGER) = l.ts
         WHERE l.dataset_id = ?
           AND l.horizon_sec IN ({placeholders})
         ORDER BY l.symbol, l.ts, l.horizon_sec
    """
    params = [dataset_id] + list(horizons)
    rows: List[DataRow] = []
    cur = conn.execute(sql, params)
    for rec in cur:
        # rec order matches select_parts
        idx = 0
        symbol = rec[idx]; idx += 1
        ts_ms = rec[idx]; idx += 1
        horizon = rec[idx]; idx += 1
        ret_bp = rec[idx]; idx += 1
        label = rec[idx]; idx += 1
        t_exec = rec[idx]; idx += 1
        uptick = rec[idx]; idx += 1
        score = rec[idx]; idx += 1
        bid = rec[idx]; idx += 1
        ask = rec[idx]; idx += 1
        vol = rec[idx] if has_vol_surge else None

        if ts_ms is None:
            continue
        try:
            ts_ms_int = int(ts_ms)
        except (TypeError, ValueError):
            continue
        ts_sec = ts_ms_int / 1000.0
        spread_bp = compute_spread_bp(_to_float(bid), _to_float(ask))
        uptick_val = _to_float(uptick)
        score_val = _to_float(score)
        vol_val = _to_float(vol)
        label_int = int(label)
        ret_bp_float = _to_float(ret_bp)
        if ret_bp_float is None:
            continue
        day = datetime.fromtimestamp(ts_sec, tz=JST).strftime("%Y%m%d")
        rows.append(
            DataRow(
                symbol=str(symbol),
                ts_ms=ts_ms_int,
                ts_sec=ts_sec,
                horizon_sec=int(horizon),
                ret_bp=float(ret_bp_float),
                label=label_int,
                uptick=uptick_val,
                score=score_val,
                spread_bp=spread_bp,
                vol_surge=vol_val,
                group=day,
            )
        )
    return rows


def _to_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def evaluate_on_indices(rows: Sequence[DataRow], indices: Sequence[int], params: Dict[str, Optional[float]]) -> Dict[str, float]:
    last_fire: Dict[str, float] = {}
    signals = 0
    hits = 0
    sum_hit = 0.0
    sum_miss = 0.0
    loss_count = 0
    sorted_idx = sorted(indices, key=lambda i: (rows[i].symbol, rows[i].ts_ms, rows[i].horizon_sec))

    uptick_thr = params["BUY_UPTICK_THR"]
    score_thr = params.get("BUY_SCORE_THR")
    spread_max = params["BUY_SPREAD_MAX"]
    cooldown = params["COOLDOWN_SEC"]
    vol_thr = params.get("VOL_SURGE_MIN")

    for idx in sorted_idx:
        row = rows[idx]
        if row.uptick is None or row.uptick < uptick_thr:
            continue
        if score_thr is not None:
            if row.score is None or row.score < score_thr:
                continue
        if row.spread_bp is None or row.spread_bp > spread_max:
            continue
        if vol_thr is not None:
            if row.vol_surge is None or row.vol_surge < vol_thr:
                continue
        last = last_fire.get(row.symbol)
        if last is not None and (row.ts_sec - last) < cooldown:
            continue
        # Signal accepted
        last_fire[row.symbol] = row.ts_sec
        signals += 1
        if row.label == 1:
            hits += 1
            sum_hit += row.ret_bp
        else:
            sum_miss += abs(row.ret_bp)
            loss_count += 1

    return {
        "signals": float(signals),
        "hits": float(hits),
        "sum_hit": sum_hit,
        "sum_miss": sum_miss,
        "loss_count": float(loss_count),
    }


def aggregate_metrics(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    total_signals = sum(m["signals"] for m in fold_metrics)
    total_hits = sum(m["hits"] for m in fold_metrics)
    sum_hit = sum(m["sum_hit"] for m in fold_metrics)
    sum_miss = sum(m["sum_miss"] for m in fold_metrics)
    loss_count = sum(m["loss_count"] for m in fold_metrics)

    if total_signals <= 0:
        precision = 0.0
    else:
        precision = total_hits / total_signals

    mean_hit = sum_hit / total_hits if total_hits > 0 else 0.0
    mean_miss = sum_miss / loss_count if loss_count > 0 else 0.0

    if total_signals <= 0:
        ev = float("-inf")
    else:
        ev = mean_hit * precision - mean_miss * (1.0 - precision)

    return {
        "signals": total_signals,
        "hits": total_hits,
        "precision": precision,
        "mean_hit": mean_hit,
        "mean_miss": mean_miss,
        "ev": ev,
    }


def _zero_metrics() -> Dict[str, float]:
    return {"signals": 0.0, "hits": 0.0, "sum_hit": 0.0, "sum_miss": 0.0, "loss_count": 0.0}


def evaluate_params(
    rows: Sequence[DataRow],
    days: Sequence[str],
    day_groups: Dict[str, List[int]],
    params: Dict[str, Optional[float]],
    cv: int,
) -> Dict[str, float]:
    if cv <= 1 or len(days) <= 1:
        indices: List[int] = []
        for day in days:
            indices.extend(day_groups.get(day, []))
        if not indices:
            return aggregate_metrics([_zero_metrics()])
        metrics = evaluate_on_indices(rows, indices, params)
        return aggregate_metrics([metrics])

    folds = make_day_folds(days, day_groups, cv)
    fold_metrics: List[Dict[str, float]] = []
    for fold_days in folds:
        indices: List[int] = []
        for day in fold_days:
            indices.extend(day_groups.get(day, []))
        if not indices:
            continue
        fold_metrics.append(evaluate_on_indices(rows, indices, params))
    if not fold_metrics:
        fold_metrics.append(_zero_metrics())
    return aggregate_metrics(fold_metrics)


def grid_parameters(has_score: bool, has_vol: bool) -> List[Dict[str, float]]:
    upticks = [0.55, 0.60, 0.65, 0.70]
    spreads = [10, 15, 20, 25, 30]
    cooldowns = [5, 10, 15]
    scores = [6.0, 6.5, 7.0] if has_score else [None]
    vols = [0.0, 0.5, 1.0] if has_vol else [None]

    params_list: List[Dict[str, float]] = []
    for u in upticks:
        for s in spreads:
            for c in cooldowns:
                for sc in scores:
                    for v in vols:
                        params: Dict[str, Optional[float]] = {
                            "BUY_UPTICK_THR": u,
                            "BUY_SPREAD_MAX": s,
                            "COOLDOWN_SEC": float(c),
                        }
                        if sc is not None:
                            params["BUY_SCORE_THR"] = sc
                        if v is not None:
                            params["VOL_SURGE_MIN"] = v
                        params_list.append(params)
    return params_list


def evaluate_grid(
    rows: Sequence[DataRow],
    days: Sequence[str],
    day_groups: Dict[str, List[int]],
    params_list: List[Dict[str, float]],
    min_trades: int,
    ev_floor: float,
    verbose: bool,
    cv: int,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    total_candidates = len(params_list)
    start = time.time()
    for idx, params in enumerate(params_list, start=1):
        agg = evaluate_params(rows, days, day_groups, params, cv)
        eligible = agg["signals"] >= min_trades and agg["ev"] >= ev_floor
        result = {
            "params": params,
            "signals": int(agg["signals"]),
            "hits": int(agg["hits"]),
            "precision": agg["precision"],
            "mean_hit": agg["mean_hit"],
            "mean_miss": agg["mean_miss"],
            "ev": agg["ev"],
            "eligible": eligible,
        }
        results.append(result)
        if verbose:
            progress = (idx / total_candidates) * 100.0
            print(
                "[grid] %4d/%d (%.1f%%) precision=%.4f trades=%d ev=%.3f params=%s"
                % (
                    idx,
                    total_candidates,
                    progress,
                    agg["precision"],
                    agg["signals"],
                    agg["ev"],
                    params,
                )
            )
    duration = time.time() - start
    if verbose:
        print(f"[grid] evaluated {total_candidates} candidates in {duration:.2f}s")
    return results


def select_top(results: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
    def sort_key(item: Dict[str, object]) -> Tuple[float, int, float]:
        return (
            float(item["precision"]),
            int(item["signals"]),
            float(item["ev"]),
        )

    eligible = [r for r in results if r["eligible"]]
    if eligible:
        ordered = sorted(eligible, key=sort_key, reverse=True)
    else:
        ordered = sorted(results, key=sort_key, reverse=True)
    return ordered[:top_k]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_csv_report(path: Path, results: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_keys = ["BUY_UPTICK_THR", "BUY_SCORE_THR", "BUY_SPREAD_MAX", "COOLDOWN_SEC", "VOL_SURGE_MIN"]
    headers = [
        "precision",
        "trades",
        "hits",
        "ev",
        "mean_hit_bp",
        "mean_loss_bp",
        "eligible",
    ] + param_keys
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\r\n")
        writer.writerow(headers)
        for row in results:
            params = row["params"]
            writer.writerow(
                [
                    "%.6f" % row["precision"],
                    row["signals"],
                    row["hits"],
                    "%.6f" % row["ev"],
                    "%.6f" % row["mean_hit"],
                    "%.6f" % row["mean_miss"],
                    int(bool(row["eligible"])),
                ]
                + [params.get(key, "") for key in param_keys]
            )


def main() -> None:
    args = parse_args()
    dataset_id = args.DatasetId
    horizons = parse_list(args.Horizons)
    min_trades = max(1, int(args.MinTrades))
    ev_floor = float(args.EVFloor)
    verbose = bool(args.Verbose)
    cv_folds = max(0, int(args.CV))

    db_path = resolve_db_path(dataset_id)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_indexes(conn)
    columns = set(feature_columns(conn))
    has_score = "score" in columns
    has_vol = "vol_surge_z" in columns
    if "f1" not in columns:
        raise SystemExit("ERROR: features_stream missing f1 (uptick) column.")

    rows = load_rows(conn, dataset_id, horizons, has_vol)
    conn.close()

    if not rows:
        raise SystemExit("ERROR: No matched rows for dataset/horizons.")

    days, day_groups = group_by_day(rows)
    params_list = grid_parameters(has_score, has_vol)
    results = evaluate_grid(rows, days, day_groups, params_list, min_trades, ev_floor, verbose, cv_folds)

    top_results = select_top(results, top_k=5)
    for idx, item in enumerate(top_results, start=1):
        params_display = {k: v for k, v in item["params"].items()}
        print(
            "[grid-top] #%d precision=%.4f trades=%d EV=%.3f params=%s"
            % (
                idx,
                item["precision"],
                item["signals"],
                item["ev"],
                params_display,
            )
        )

    best = top_results[0] if top_results else results[0]
    best_is_eligible = bool(best["eligible"])
    best_params = {k: v for k, v in best["params"].items() if v is not None}
    best_payload = {
        "dataset_id": dataset_id,
        "precision": best["precision"],
        "trades": best["signals"],
        "ev": best["ev"],
        "mean_hit_bp": best["mean_hit"],
        "mean_loss_bp": best["mean_miss"],
        "params": best_params,
        "eligible": best_is_eligible,
    }

    out_path = Path(args.Out) if args.Out else Path(f"exports/best_thresholds_{dataset_id}.json")
    write_json(out_path, best_payload)

    report_path = Path(f"exports/grid_report_{dataset_id}.csv")
    write_csv_report(report_path, results)

    print(f"[grid] best saved to {out_path}")
    print(f"[grid] report saved to {report_path}")
    if not top_results or not best_is_eligible:
        print("[grid] warning: no parameter set satisfied MinTrades/EVFloor constraints.")


if __name__ == "__main__":
    main()
