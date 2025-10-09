# -*- coding: utf-8 -*-
"""
SELL-side Grid Search for threshold optimization.

This script mirrors the BUY-side grid search but evaluates SELL entries.
Key differences:
- Outcome labels/returns are flipped to make "down = win" map to label=1 and positive bp.
- Output files are SELL-specific: best_thresholds_sell_<dataset>.json, grid_report_sell_<dataset>.csv

Usage (PowerShell):
  py -m scripts.grid_search_thresholds_sell -DatasetId REF20251008_0828 -MinTrades 50 -EVFloor -50 -Verbose 1
"""
import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None

JST = ZoneInfo("Asia/Tokyo") if ZoneInfo else timezone.utc

# -------------------------
# Data structures
# -------------------------

@dataclass
class DataRow:
    symbol: str
    ts_ms: int
    ts_sec: float
    day: str
    horizon_sec: int
    # SELL-normalized outcome:
    #  - ret_bp: positive means favorable for SELL (i.e., price went DOWN)
    #  - label: 1 means win (down enough), 0 means loss (up enough)
    ret_bp: float
    label: int
    # features (available columns may vary)
    spread_bp: Optional[float] = None
    uptick_ratio_10t: Optional[float] = None
    score: Optional[float] = None
    vol_surge_z: Optional[float] = None


# -------------------------
# Helpers
# -------------------------

def resolve_db_path(dataset_id: str) -> Path:
    """Find the dated refeed DB path from dataset_registry rows in db/naut_market_*_refeed.db."""
    db_dir = Path("db")
    candidates = sorted(db_dir.glob("naut_market_*_refeed.db"))
    if not candidates:
        raise SystemExit("No refeed DBs found under db\\*. Run stream_microbatch first.")
    for path in reversed(candidates):
        con = sqlite3.connect(str(path))
        try:
            cur = con.execute(
                "SELECT source_db_path FROM dataset_registry WHERE dataset_id=? LIMIT 1",
                (dataset_id,)
            )
            row = cur.fetchone()
            if row:
                return path  # This DB contains the requested dataset_id
        finally:
            con.close()
    raise SystemExit(f"DatasetId {dataset_id} not found in any refeed DB.")


def fetch_available_columns(con: sqlite3.Connection) -> List[str]:
    cols = []
    cur = con.execute("PRAGMA table_info(features_stream)")
    for _, name, *_ in cur.fetchall():
        cols.append(name)
    return cols


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_rows(con: sqlite3.Connection, horizons: List[int], dataset_id: str) -> Tuple[List[DataRow], Dict[str, bool]]:
    """
    Join features_stream and labels_outcome for the given dataset.
    SELL-normalize outcomes: ret_bp *= -1, label = 1 - label.
    """
    # check columns we might use
    fcols = fetch_available_columns(con)
    has_spread = "spread_bp" in fcols
    # uptick ratio candidates (common names)
    uptick_candidates = [c for c in ("uptick_ratio_10t", "uptick_ratio", "uptick") if c in fcols]
    has_uptick = bool(uptick_candidates)
    uptick_col = uptick_candidates[0] if uptick_candidates else None
    has_score = "score" in fcols
    has_volsurge = "vol_surge_z" in fcols

    # Build dynamic SELECT for features
    sel_feats = ["f.symbol", "CAST(f.t_exec*1000 AS INTEGER) AS ts_ms", "f.t_exec AS ts_sec"]
    if has_spread: sel_feats.append("f.spread_bp")
    if has_uptick: sel_feats.append(f"f.{uptick_col} AS uptick_ratio_10t")
    if has_score: sel_feats.append("f.score")
    if has_volsurge: sel_feats.append("f.vol_surge_z")

    sel_feats_sql = ", ".join(sel_feats)

    # Labels table rows for all horizons
    # We SELECT (symbol, ts, horizon_sec, ret_bp, label) and join to features by timestamp.
    rows: List[DataRow] = []
    for h in horizons:
        sql = f"""
        SELECT {sel_feats_sql}, l.horizon_sec, l.ret_bp, l.label
          FROM features_stream f
          JOIN labels_outcome l
            ON CAST(f.t_exec*1000 AS INTEGER) = l.ts
         WHERE l.dataset_id = ?
           AND l.horizon_sec = ?
        """
        for rec in con.execute(sql, (dataset_id, h)):
            idx = 0
            symbol = rec[idx]; idx += 1
            ts_ms = int(rec[idx]); idx += 1
            ts_sec = float(rec[idx]); idx += 1

            spread_bp = uptick = score = volsurge = None
            if has_spread:
                spread_bp = _to_float(rec[idx]); idx += 1
            if has_uptick:
                uptick = _to_float(rec[idx]); idx += 1
            if has_score:
                score = _to_float(rec[idx]); idx += 1
            if has_volsurge:
                volsurge = _to_float(rec[idx]); idx += 1

            horizon_sec = int(rec[idx]); idx += 1
            ret_bp = _to_float(rec[idx]); idx += 1
            label = int(rec[idx]); idx += 1

            # SELL-normalization
            # Make "favorable for SELL" positive bp and label=1
            if ret_bp is None:
                continue
            ret_bp = -ret_bp        # in SELL, price going DOWN is profit => positive bp after flip
            label = 1 - label       # in SELL, "down enough" should be label=1

            dt = datetime.fromtimestamp(ts_sec, tz=JST)
            day = dt.strftime("%Y%m%d")

            rows.append(DataRow(
                symbol=str(symbol),
                ts_ms=ts_ms,
                ts_sec=ts_sec,
                day=day,
                horizon_sec=horizon_sec,
                ret_bp=float(ret_bp),
                label=int(label),
                spread_bp=spread_bp,
                uptick_ratio_10t=uptick,
                score=score,
                vol_surge_z=volsurge
            ))
    # Feature availability flags
    avail = {
        "spread_bp": has_spread,
        "uptick_ratio_10t": has_uptick,
        "score": has_score,
        "vol_surge_z": has_volsurge,
    }
    return rows, avail


def group_by_day(rows: Sequence[DataRow]) -> Tuple[List[str], Dict[str, List[int]]]:
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.day, []).append(i)
    days = sorted(groups.keys())
    return days, groups


def simulate_signals(rows: Sequence[DataRow],
                     params: Dict[str, float],
                     cooldown_sec: float,
                     avail: Dict[str, bool],
                     candidate_indices: Optional[Sequence[int]] = None) -> List[int]:
    """
    Return indices of rows where a SELL entry would fire under the parameter set.
    We require features when available; if a feature is unavailable, that condition is ignored.
    """
    buy_uptick_thr = params.get("SELL_UPTICK_THR", None)
    spread_max = params.get("SELL_SPREAD_MAX", None)
    score_thr = params.get("SELL_SCORE_THR", None)

    last_fire_by_symbol: Dict[str, float] = {}
    fired: List[int] = []

    if candidate_indices is None:
        iter_indices = range(len(rows))
    else:
        iter_indices = sorted(candidate_indices, key=lambda idx: (rows[idx].symbol, rows[idx].ts_ms, rows[idx].horizon_sec))

    for i in iter_indices:
        r = rows[i]
        # cooldown per symbol
        last = last_fire_by_symbol.get(r.symbol, -1e18)
        if r.ts_sec - last < cooldown_sec:
            continue

        ok = True

        # For SELL, we'd like "down-tick" dominance; many feeds only provide "uptick ratio".
        # A simple heuristic: when available, require (1 - uptick_ratio) >= threshold.
        if buy_uptick_thr is not None and avail.get("uptick_ratio_10t", False) and r.uptick_ratio_10t is not None:
            down_ratio = 1.0 - r.uptick_ratio_10t
            if down_ratio < buy_uptick_thr:
                ok = False

        if ok and spread_max is not None and avail.get("spread_bp", False) and r.spread_bp is not None:
            if r.spread_bp > spread_max:
                ok = False

        if ok and score_thr is not None and avail.get("score", False) and r.score is not None:
            if r.score < score_thr:
                ok = False

        if ok:
            fired.append(i)
            last_fire_by_symbol[r.symbol] = r.ts_sec

    return fired


def _zero_metrics() -> Dict[str, float]:
    return {"signals": 0.0, "hits": 0.0, "sum_hit": 0.0, "sum_miss": 0.0, "loss_count": 0.0}


def aggregate_metrics(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    total_signals = sum(m["signals"] for m in fold_metrics)
    total_hits = sum(m["hits"] for m in fold_metrics)
    sum_hit = sum(m["sum_hit"] for m in fold_metrics)
    sum_miss = sum(m["sum_miss"] for m in fold_metrics)
    loss_count = sum(m["loss_count"] for m in fold_metrics)

    precision = total_hits / total_signals if total_signals > 0 else 0.0
    mean_hit = sum_hit / total_hits if total_hits > 0 else 0.0
    mean_miss = sum_miss / loss_count if loss_count > 0 else 0.0
    ev = mean_hit * precision - mean_miss * (1 - precision) if total_signals > 0 else float("-inf")
    return {
        "signals": total_signals,
        "hits": total_hits,
        "precision": precision,
        "mean_hit": mean_hit,
        "mean_miss": mean_miss,
        "ev": ev,
    }


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


def evaluate_on_indices(
    rows: Sequence[DataRow],
    indices: Sequence[int],
    params: Dict[str, float],
    cooldown_sec: float,
    avail: Dict[str, bool],
) -> Dict[str, float]:
    metrics = _zero_metrics()
    fired_idx = simulate_signals(rows, params, cooldown_sec, avail, candidate_indices=indices)
    for idx in fired_idx:
        metrics["signals"] += 1.0
        row = rows[idx]
        if row.label == 1:
            metrics["hits"] += 1.0
            metrics["sum_hit"] += row.ret_bp
        else:
            metrics["sum_miss"] += abs(row.ret_bp)
            metrics["loss_count"] += 1.0
    return metrics


def evaluate_params(
    rows: Sequence[DataRow],
    days: Sequence[str],
    day_groups: Dict[str, List[int]],
    params: Dict[str, float],
    cooldown_sec: float,
    cv: int,
    avail: Dict[str, bool],
) -> Dict[str, float]:
    if cv <= 1 or len(days) <= 1:
        indices: List[int] = []
        for day in days:
            indices.extend(day_groups.get(day, []))
        if not indices:
            return aggregate_metrics([_zero_metrics()])
        metrics = evaluate_on_indices(rows, indices, params, cooldown_sec, avail)
        return aggregate_metrics([metrics])

    folds = make_day_folds(days, day_groups, cv)
    fold_metrics: List[Dict[str, float]] = []
    for fold_days in folds:
        indices: List[int] = []
        for day in fold_days:
            indices.extend(day_groups.get(day, []))
        if not indices:
            continue
        fold_metrics.append(evaluate_on_indices(rows, indices, params, cooldown_sec, avail))
    if not fold_metrics:
        fold_metrics.append(_zero_metrics())
    return aggregate_metrics(fold_metrics)


def run_grid(rows: Sequence[DataRow],
             days: Sequence[str],
             day_groups: Dict[str, List[int]],
             avail: Dict[str, bool],
             min_trades: int,
             ev_floor: float,
             cv: int = 0,
             verbose: int = 0):
    # SELL-oriented grid (a bit looser on "down dominance")
    grid_uptick = [0.30, 0.40, 0.50, 0.60]     # threshold for (1 - uptick_ratio)
    grid_spread = [10, 15, 20, 25, 30]         # bp
    grid_score = [5.5, 6.0, 6.5, 7.0]          # if score exists
    grid_cooldown = [5, 10, 15]                # seconds

    results = []
    total = len(grid_uptick) * len(grid_spread) * len(grid_score) * len(grid_cooldown)
    c = 0
    for up in grid_uptick:
        for sp in grid_spread:
            for sc in grid_score:
                for cd in grid_cooldown:
                    c += 1
                    params = {
                        "SELL_UPTICK_THR": up,
                        "SELL_SPREAD_MAX": sp,
                        "SELL_SCORE_THR": sc,
                    }
                    metrics = evaluate_params(rows, days, day_groups, params, cd, cv, avail)
                    trades = int(metrics["signals"])
                    precision = metrics["precision"]
                    ev = metrics["ev"]
                    eligible = (trades >= min_trades) and (ev >= ev_floor)
                    results.append({
                        "precision": precision,
                        "trades": trades,
                        "EV": ev,
                        "eligible": eligible,
                        "params": {**params, "COOLDOWN_SEC": cd},
                    })
                    if verbose:
                        print(f"[{c}/{total}] up={up} sp={sp} sc={sc} cd={cd} -> P={precision:.4f} N={trades} EV={ev:.2f} eligible={eligible}")
    # Sort: precision desc, then trades desc, then EV desc
    results.sort(key=lambda r: (r["precision"], r["trades"], r["EV"]), reverse=True)
    return results


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["precision","trades","EV","eligible","params"])
        return
    # flatten params for convenience
    keys = ["precision","trades","EV","eligible","SELL_UPTICK_THR","SELL_SPREAD_MAX","SELL_SCORE_THR","COOLDOWN_SEC"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(keys)
        for r in rows:
            p = r["params"]
            w.writerow([
                f"{r['precision']:.6f}", r["trades"], f"{r['EV']:.4f}", r["eligible"],
                p.get("SELL_UPTICK_THR"), p.get("SELL_SPREAD_MAX"), p.get("SELL_SCORE_THR"), p.get("COOLDOWN_SEC"),
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-DatasetId", required=True, help="Target dataset id (e.g., REF20251008_0828)")
    ap.add_argument("-Horizons", default="60,120", help="Comma separated horizons in seconds (default: 60,120)")
    ap.add_argument("-MinTrades", type=int, default=50, help="Minimum trades to be eligible (default: 50)")
    ap.add_argument("-EVFloor", type=float, default=-50.0, help="Minimum EV(bp) to be eligible (default: -50)")
    ap.add_argument("-Out", default=None, help="Path to write best JSON (default: exports/best_thresholds_sell_<dataset>.json)")
    ap.add_argument("-CV", type=int, default=0, help="Number of day-based CV folds (0 disables CV)")
    ap.add_argument("-Verbose", type=int, default=1)
    args = ap.parse_args()

    horizons = [int(x) for x in str(args.Horizons).split(",") if x.strip()]
    if not horizons:
        raise SystemExit("No horizons provided.")

    db_path = resolve_db_path(args.DatasetId)
    con = sqlite3.connect(str(db_path))

    rows, avail = load_rows(con, horizons, args.DatasetId)
    con.close()

    if not rows:
        raise SystemExit("No joined rows found. Ensure labels_outcome and features_stream are populated.")

    days, day_groups = group_by_day(rows)
    cv_folds = max(0, int(args.CV))

    results = run_grid(
        rows,
        days,
        day_groups,
        avail,
        min_trades=args.MinTrades,
        ev_floor=args.EVFloor,
        cv=cv_folds,
        verbose=args.Verbose,
    )

    top = results[:5]
    for i, r in enumerate(top, 1):
        print(f"[grid-top] #{i} precision={r['precision']:.4f} trades={r['trades']} EV={r['EV']:.3f} params={r['params']}")

    dataset_id = args.DatasetId
    out_json = Path(args.Out) if args.Out else Path(f"exports/best_thresholds_sell_{dataset_id}.json")
    save_json(out_json, results[0] if results else {"eligible": False, "params": {}, "precision": 0.0, "trades": 0, "EV": float("nan")})
    print(f"[grid] best saved to {out_json}")

    out_csv = Path(f"exports/grid_report_sell_{dataset_id}.csv")
    save_csv(out_csv, results)
    print(f"[grid] report saved to {out_csv}")

if __name__ == "__main__":
    main()
