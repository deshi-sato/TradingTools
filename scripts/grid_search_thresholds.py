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
import hashlib
import json
import math
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import pandas as pd

JST = timezone(timedelta(hours=9))

CACHE_VERSION = 1
CACHE_DIR = Path("out/grid_cache")


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
    spread_ticks: Optional[float]
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
    parser.add_argument("-Symbols", help="Comma-separated list of symbols to include")
    parser.add_argument("-Verbose", type=int, default=0)
    parser.add_argument(
        "-ProgressStep",
        type=int,
        default=500,
        help="Progress output interval (number of candidates, default: 500).",
    )
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


def parse_symbols_arg(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        return None
    seen = set()
    ordered: List[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            ordered.append(part)
    return ordered


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


def load_rows(
    conn: sqlite3.Connection,
    dataset_id: str,
    horizons: Sequence[int],
    has_vol_surge: bool,
    use_ts_ms: bool,
    symbols: Optional[Sequence[str]] = None,
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
    ]
    if has_vol_surge:
        select_parts.append("f.vol_surge_z AS vol_surge_z")
    join_condition = "f.symbol = l.symbol"
    if use_ts_ms:
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
    if symbols:
        symbols_clean = [s.strip() for s in symbols if s.strip()]
        symbols_unique: List[str] = []
        seen_symbols = set()
        for sym in symbols_clean:
            if sym not in seen_symbols:
                seen_symbols.add(sym)
                symbols_unique.append(sym)
        if symbols_unique:
            symbol_placeholders = ",".join(["?"] * len(symbols_unique))
            sql += f" AND l.symbol IN ({symbol_placeholders})"
            params.extend(symbols_unique)
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
    numeric_optional = ["f1", "score", "spread_ticks"]
    if has_vol_surge:
        numeric_optional.append("vol_surge_z")
    for col in numeric_optional:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    ts_jst = pd.to_datetime(df["ts_sec"], unit="s") + pd.to_timedelta(9, unit="h")
    df["group"] = ts_jst.dt.strftime("%Y%m%d")
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "ts_ms", "horizon_sec"]).reset_index(drop=True)

    def _opt(val: Any) -> Optional[float]:
        return None if pd.isna(val) else float(val)

    rows: List[DataRow] = []
    for row in df.itertuples(index=False, name="Row"):
        rows.append(
            DataRow(
                symbol=row.symbol,
                ts_ms=int(row.ts_ms),
                ts_sec=float(row.ts_sec),
                horizon_sec=int(row.horizon_sec),
                ret_bp=float(row.ret_bp),
                label=int(row.label),
                uptick=_opt(getattr(row, "f1", None)),
                score=_opt(getattr(row, "score", None)),
                spread_ticks=_opt(getattr(row, "spread_ticks", None)),
                vol_surge=_opt(getattr(row, "vol_surge_z", None))
                if hasattr(row, "vol_surge_z")
                else None,
                group=row.group,
            )
        )
    return rows


def group_by_day(rows: Sequence[DataRow]) -> Tuple[List[str], Dict[str, List[int]]]:
    day_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        day_groups[row.group].append(idx)
    days = sorted(day_groups.keys())
    return days, day_groups


def _dataset_signature(path: Path) -> str:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return ""
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _build_cache_key(
    dataset_id: str,
    horizons: Sequence[int],
    has_vol: bool,
    has_ts_ms: bool,
    db_signature: str,
    symbols: Optional[Sequence[str]],
) -> str:
    payload = {
        "dataset_id": dataset_id,
        "horizons": list(horizons),
        "has_vol": has_vol,
        "has_ts_ms": has_ts_ms,
        "db_signature": db_signature,
        "symbols": list(symbols) if symbols else [],
        "version": CACHE_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def load_dataset_cache(
    dataset_id: str,
    horizons: Sequence[int],
    has_vol: bool,
    has_ts_ms: bool,
    db_signature: str,
    symbols: Optional[Sequence[str]],
) -> Optional[Tuple[List[DataRow], List[str], Dict[str, List[int]]]]:
    key = _build_cache_key(dataset_id, horizons, has_vol, has_ts_ms, db_signature, symbols)
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            payload: Dict[str, Any] = pickle.load(fh)
    except (OSError, pickle.UnpicklingError):
        return None
    meta = payload.get("meta", {})
    if meta.get("version") != CACHE_VERSION:
        return None
    rows = payload.get("rows")
    days = payload.get("days")
    day_groups = payload.get("day_groups")
    if rows is None or days is None or day_groups is None:
        return None
    return rows, days, day_groups


def save_dataset_cache(
    dataset_id: str,
    horizons: Sequence[int],
    has_vol: bool,
    has_ts_ms: bool,
    db_signature: str,
    symbols: Optional[Sequence[str]],
    rows: Sequence[DataRow],
    days: Sequence[str],
    day_groups: Dict[str, List[int]],
) -> None:
    key = _build_cache_key(dataset_id, horizons, has_vol, has_ts_ms, db_signature, symbols)
    path = _cache_path(key)
    payload = {
        "meta": {
            "version": CACHE_VERSION,
            "dataset_id": dataset_id,
            "horizons": list(horizons),
            "has_vol": has_vol,
            "has_ts_ms": has_ts_ms,
            "db_signature": db_signature,
            "symbols": list(symbols) if symbols else [],
            "row_count": len(rows),
        },
        "rows": list(rows),
        "days": list(days),
        "day_groups": day_groups,
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


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
        if row.spread_ticks is None or row.spread_ticks > spread_max:
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
    upticks = [round(0.40 + 0.05 * i, 2) for i in range(7)]
    spreads = [10, 20, 30]
    cooldowns = [5.0, 10.0, 15.0]
    scores = [round(5.0 + 0.5 * i, 1) for i in range(7)] if has_score else [None]
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
    progress_step: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    results: List[Dict[str, object]] = []
    eligible_results: List[Dict[str, object]] = []
    total_candidates = len(params_list)
    step = max(1, progress_step)
    start = time.time()

    def result_rank_key(item: Dict[str, object]) -> Tuple[int, float, int, float]:
        eligible_score = 1 if item["eligible"] else 0
        return (
            eligible_score,
            float(item["precision"]),
            int(item["signals"]),
            float(item["ev"]),
        )

    best_so_far: Optional[Dict[str, object]] = None
    best_eligible: Optional[Dict[str, object]] = None

    print(f"[grid] total candidates = {total_candidates}")
    sys.stdout.flush()

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
        if eligible:
            eligible_results.append(result)
        if best_so_far is None or result_rank_key(result) > result_rank_key(best_so_far):
            best_so_far = result
        if eligible:
            if best_eligible is None or result_rank_key(result) > result_rank_key(best_eligible):
                best_eligible = result
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
        if (idx % step == 0) or (idx == total_candidates):
            elapsed = time.time() - start
            cps = idx / elapsed if elapsed > 0 else 0.0
            pct = (idx / total_candidates * 100.0) if total_candidates else 100.0
            eta = ((total_candidates - idx) / cps) if cps > 0 else 0.0
            if best_eligible:
                be = best_eligible
                best_ev = float(be["ev"]) if be["ev"] is not None else 0.0
                best_prec = float(be["precision"]) if be["precision"] is not None else 0.0
                best_trades = int(be["signals"]) if be["signals"] is not None else 0
                best_str = f"best(ev={best_ev:.3f}, prec={best_prec:.3f}, trades={best_trades})"
            elif best_so_far:
                bs = best_so_far
                best_ev = float(bs["ev"]) if bs["ev"] is not None else 0.0
                best_prec = float(bs["precision"]) if bs["precision"] is not None else 0.0
                best_trades = int(bs["signals"]) if bs["signals"] is not None else 0
                best_str = f"best-all(ev={best_ev:.3f}, prec={best_prec:.3f}, trades={best_trades})"
            else:
                best_str = "best n/a"
            print(
                "[grid] {done:,}/{total:,} ({pct:4.1f}%)  {cps:,.0f}/s  "
                "elapsed={elapsed:,.1f}s  ETA~{eta:,.1f}s  eligible={eligible:,}  {best}".format(
                    done=idx,
                    total=total_candidates,
                    pct=pct,
                    cps=cps,
                    elapsed=elapsed,
                    eta=eta,
                    eligible=len(eligible_results),
                    best=best_str,
                )
            )
            sys.stdout.flush()
    duration = time.time() - start
    print(f"[grid] evaluated {total_candidates} candidates in {duration:.2f}s")
    print(f"[grid] eligible candidates = {len(eligible_results)}/{total_candidates}")
    return results, eligible_results


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


def write_candidates_csv(path: Path, results: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "BUY_UPTICK_THR",
        "BUY_SCORE_THR",
        "BUY_SPREAD_MAX",
        "COOLDOWN_SEC",
        "precision",
        "ev",
        "trades",
        "mean_hit_bp",
        "mean_loss_bp",
        "eligible",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\r\n")
        writer.writerow(headers)
        for row in results:
            params = row["params"]
            writer.writerow(
                [
                    params.get("BUY_UPTICK_THR", ""),
                    params.get("BUY_SCORE_THR", ""),
                    params.get("BUY_SPREAD_MAX", ""),
                    params.get("COOLDOWN_SEC", ""),
                    row.get("precision"),
                    row.get("ev"),
                    row.get("signals"),
                    row.get("mean_hit"),
                    row.get("mean_miss"),
                    int(bool(row.get("eligible"))),
                ]
            )


def main() -> None:
    args = parse_args()
    dataset_id = args.DatasetId
    horizons = parse_list(args.Horizons)
    symbols_filter = parse_symbols_arg(getattr(args, "Symbols", None))
    min_trades = max(1, int(args.MinTrades))
    ev_floor = float(args.EVFloor)
    verbose = bool(args.Verbose)
    cv_folds = max(0, int(args.CV))

    db_path = resolve_db_path(dataset_id)
    db_signature = _dataset_signature(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    columns = set(feature_columns(conn))
    has_ts_ms = "ts_ms" in columns
    ensure_indexes(conn, has_ts_ms)
    has_score = "score" in columns
    has_vol = "vol_surge_z" in columns
    if "f1" not in columns:
        raise SystemExit("ERROR: features_stream missing f1 (uptick) column.")

    cache_hit = load_dataset_cache(dataset_id, horizons, has_vol, has_ts_ms, db_signature, symbols_filter)
    if cache_hit:
        rows, days, day_groups = cache_hit
        print(f"[grid] dataset cache hit (rows={len(rows)})")
    else:
        rows = load_rows(conn, dataset_id, horizons, has_vol, has_ts_ms, symbols_filter)
        if not rows:
            conn.close()
            raise SystemExit("ERROR: No matched rows for dataset/horizons.")
        days, day_groups = group_by_day(rows)
        save_dataset_cache(
            dataset_id,
            horizons,
            has_vol,
            has_ts_ms,
            db_signature,
            symbols_filter,
            rows,
            days,
            day_groups,
        )
        print(f"[grid] dataset cache stored (rows={len(rows)})")
    conn.close()

    params_list = grid_parameters(has_score, has_vol)
    if symbols_filter:
        print(f"[grid] symbol filter: {', '.join(symbols_filter)}")
    print(f"[grid] applying filters: MinTrades>={min_trades:,}, EVFloor>={ev_floor:.3f}")
    all_results, eligible_results = evaluate_grid(
        rows,
        days,
        day_groups,
        params_list,
        min_trades,
        ev_floor,
        verbose,
        cv_folds,
        max(1, int(args.ProgressStep)),
    )

    key_fn = lambda r: (float(r["precision"]), float(r["ev"]), int(r["signals"]))

    filtered_results = eligible_results if eligible_results else all_results

    top_results = select_top(filtered_results, top_k=5)
    for idx, item in enumerate(top_results, start=1):
        params_display = {k: v for k, v in item["params"].items()}
        eligible_flag = "PASS" if item.get("eligible") else "FAIL"
        print(
            "[grid-top] #%d precision=%.4f trades=%d EV=%.3f eligible=%s params=%s"
            % (
                idx,
                item["precision"],
                item["signals"],
                item["ev"],
                eligible_flag,
                params_display,
            )
        )

    if eligible_results:
        best = max(eligible_results, key=key_fn)
        best_is_eligible = True
    else:
        best = max(all_results, key=key_fn)
        best_is_eligible = False
        print("[grid] warning: no parameter set satisfied MinTrades/EVFloor constraints.")
        print(
            "[grid] selecting best among all candidates for reference: "
            f"precision={best['precision']:.3f} trades={best['signals']} ev={best['ev']:.3f}"
        )

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
        "min_trades": min_trades,
        "ev_floor": ev_floor,
    }
    if not best_is_eligible:
        best_payload["ineligible_reason"] = (
            f"signals={best['signals']} (need >={min_trades}), "
            f"ev={best['ev']:.3f} (need >={ev_floor})"
        )

    if args.Out:
        out_arg = Path(args.Out)
        if out_arg.suffix.lower() == ".json":
            out_path = out_arg
        else:
            out_path = out_arg / f"best_thresholds_{dataset_id}.json"
    else:
        out_path = Path(f"exports/best_thresholds_{dataset_id}.json")

    write_json(out_path, best_payload)

    report_path = Path(f"exports/grid_report_{dataset_id}.csv")
    write_csv_report(report_path, all_results)
    filtered_report_path: Optional[Path] = None
    if eligible_results and len(eligible_results) != len(all_results):
        filtered_report_path = Path(f"exports/grid_report_eligible_{dataset_id}.csv")
        write_csv_report(filtered_report_path, eligible_results)

    candidates_path = out_path.parent / f"grid_candidates_{dataset_id}.csv"
    write_candidates_csv(candidates_path, filtered_results)
    all_candidates_path = out_path.parent / f"grid_candidates_all_{dataset_id}.csv"
    write_candidates_csv(all_candidates_path, all_results)

    print(f"[grid] best saved to {out_path}")
    print(f"[grid] report saved to {report_path}")
    if filtered_report_path:
        print(f"[grid] eligible report saved to {filtered_report_path}")
    print(f"[grid] candidates saved to {candidates_path}")
    print(f"[grid] all candidates saved to {all_candidates_path}")
    if not top_results or not best_is_eligible:
        print("[grid] warning: no parameter set satisfied MinTrades/EVFloor constraints.")


if __name__ == "__main__":
    main()

