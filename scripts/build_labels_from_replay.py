#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_labels_from_replay.py

Generates supervised labels (labels_outcome) from events_replay and features_stream.
Labels capture whether price moved up/down beyond given bp thresholds within specified horizons.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

JST = timezone(timedelta(hours=9))

LABELS_DDL = """
CREATE TABLE IF NOT EXISTS labels_outcome (
  symbol TEXT NOT NULL,
  ts     INTEGER NOT NULL,
  horizon_sec INTEGER NOT NULL,
  ret_bp REAL NOT NULL,
  label  INTEGER NOT NULL,
  dataset_id TEXT NOT NULL,
  PRIMARY KEY (symbol, ts, horizon_sec, dataset_id)
);
"""


def get_jst_now() -> datetime:
    return datetime.now(tz=JST)


def load_dataset_meta(db_path: Path, dataset_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT * FROM dataset_registry WHERE dataset_id=?",
            (dataset_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"ERROR: dataset_id {dataset_id} not found in dataset_registry.")
        return row
    finally:
        conn.close()


def resolve_db_path(dataset_id: str) -> Path:
    db_dir = Path("db")
    candidates = sorted(db_dir.glob("naut_market_*_refeed.db"))
    if not candidates:
        raise SystemExit("ERROR: no refeed DBs found under db/")
    for path in candidates:
        conn = sqlite3.connect(str(path))
        try:
            cur = conn.execute(
                "SELECT source_db_path FROM dataset_registry WHERE dataset_id=? LIMIT 1",
                (dataset_id,),
            )
            row = cur.fetchone()
            if row:
                source = row[0]
                final_path = Path(source) if source else path
                if not final_path.exists():
                    final_path = path
                return final_path.resolve()
        finally:
            conn.close()
    raise SystemExit(f"ERROR: dataset_id {dataset_id} not found in any refeed DB.")


def parse_horizons(horizons: str) -> List[int]:
    result: List[int] = []
    for part in horizons.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise SystemExit(f"ERROR: invalid horizon value '{part}'")
        if value <= 0:
            raise SystemExit(f"ERROR: horizon must be positive (got {value})")
        result.append(value)
    if not result:
        raise SystemExit("ERROR: at least one horizon required")
    return sorted(set(result))


def parse_thresholds(thresholds: str) -> Tuple[float, float]:
    parts = [p.strip() for p in thresholds.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit("ERROR: -Thresholds requires two comma-separated values (up,down)")
    try:
        up = float(parts[0])
        down = float(parts[1])
    except ValueError:
        raise SystemExit("ERROR: thresholds must be numeric")
    if down >= 0:
        raise SystemExit("ERROR: negative threshold must be <0 for down side")
    if up <= 0:
        raise SystemExit("ERROR: positive threshold must be >0 for up side")
    return up, down


def parse_bp_mode(mode: str) -> str:
    allowed = {"mid"}
    mode_lower = mode.lower()
    if mode_lower not in allowed:
        raise SystemExit(f"ERROR: unsupported bp mode '{mode}'. allowed: {sorted(allowed)}")
    return mode_lower


@dataclass
class FeatureRow:
    symbol: str
    ts: float
    bid1: Optional[float]
    ask1: Optional[float]


def fetch_features(conn: sqlite3.Connection, dataset_id: str) -> Dict[str, List[FeatureRow]]:
    cur = conn.execute(
        """
        SELECT symbol, t_exec, bid1, ask1
          FROM features_stream
         WHERE t_exec IS NOT NULL
         ORDER BY symbol ASC, t_exec ASC
        """
    )
    result: Dict[str, List[FeatureRow]] = {}
    for symbol, ts, bid, ask in cur:
        try:
            ts_float = float(ts)
        except (TypeError, ValueError):
            continue
        rows = result.setdefault(symbol, [])
        rows.append(FeatureRow(symbol, ts_float, _to_float(bid), _to_float(ask)))
    return result


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mid_price(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


def compute_labels_for_symbol(
    symbol: str,
    rows: Sequence[FeatureRow],
    horizons: Sequence[int],
    up_thr: float,
    down_thr: float,
) -> Tuple[List[Tuple[str, int, int, float, int]], Dict[str, int]]:
    inserted: List[Tuple[str, int, int, float, int]] = []
    stats = {"skipped_mid": 0, "skipped_future": 0, "total": 0, "inserted": 0}
    if not rows:
        return inserted, stats

    ts_list = [row.ts for row in rows]
    for idx, row in enumerate(rows):
        current_mid = _mid_price(row.bid1, row.ask1)
        if current_mid is None or current_mid <= 0:
            stats["skipped_mid"] += 1
            continue
        for horizon in horizons:
            target_ts = row.ts + horizon
            future_idx = _find_future_index(ts_list, target_ts, idx + 1)
            if future_idx is None:
                stats["skipped_future"] += 1
                continue
            future_row = rows[future_idx]
            future_mid = _mid_price(future_row.bid1, future_row.ask1)
            if future_mid is None or future_mid <= 0:
                stats["skipped_future"] += 1
                continue
            ret_bp = ((future_mid - current_mid) / current_mid) * 10000.0
            stats["total"] += 1
            label: Optional[int]
            if ret_bp >= up_thr:
                label = 1
            elif ret_bp <= down_thr:
                label = 0
            else:
                label = None
            if label is None:
                continue
            stats["inserted"] += 1
            inserted.append(
                (
                    symbol,
                    int(round(row.ts * 1000)),
                    horizon,
                    ret_bp,
                    label,
                )
            )
    return inserted, stats


def _find_future_index(ts_list: Sequence[float], target: float, start_idx: int) -> Optional[int]:
    lo = start_idx
    hi = len(ts_list) - 1
    if lo >= len(ts_list):
        return None
    while lo <= hi:
        mid = (lo + hi) // 2
        if ts_list[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo < len(ts_list):
        return lo
    return None


def insert_labels(
    conn: sqlite3.Connection,
    dataset_id: str,
    entries: Iterable[Tuple[str, int, int, float, int]],
) -> int:
    if not entries:
        return 0
    rows = [
        (symbol, ts_ms, horizon, ret_bp, label, dataset_id)
        for symbol, ts_ms, horizon, ret_bp, label in entries
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO labels_outcome(symbol, ts, horizon_sec, ret_bp, label, dataset_id)
        VALUES (?,?,?,?,?,?)
        """,
        rows,
    )
    return len(rows)


def run_job(args: argparse.Namespace) -> None:
    dataset_id = args.DatasetId
    horizons = parse_horizons(args.Horizons or "60,120")
    up_thr, down_thr = parse_thresholds(args.Thresholds or "+8,-6")
    bp_mode = parse_bp_mode(args.BpMode or "mid")
    if bp_mode != "mid":
        raise SystemExit("ERROR: only mid mode currently implemented")

    db_path = resolve_db_path(dataset_id)
    dataset_meta = load_dataset_meta(db_path, dataset_id)
    print("===== build_labels_from_replay =====")
    print(f"[labels] dataset_id={dataset_id}")
    print(f"[labels] db_path={db_path}")
    print(f"[labels] horizons={horizons}")
    print(f"[labels] thresholds(up,down)=({up_thr},{down_thr})")
    print(f"[labels] bp_mode={bp_mode}")
    if dataset_meta is not None:
        created_at = dataset_meta["created_at"] if "created_at" in dataset_meta.keys() else None
        print(f"[labels] dataset_created_at={created_at}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(LABELS_DDL)

    feature_rows = fetch_features(conn, dataset_id)
    total_inserted = 0
    total_skipped_mid = 0
    total_skipped_future = 0
    total_considered = 0

    start_time = time.time()
    for symbol, rows in feature_rows.items():
        entries, stats = compute_labels_for_symbol(symbol, rows, horizons, up_thr, down_thr)
        inserted = insert_labels(conn, dataset_id, entries)
        total_inserted += inserted
        total_skipped_mid += stats["skipped_mid"]
        total_skipped_future += stats["skipped_future"]
        total_considered += stats["total"]
    conn.commit()

    elapsed = time.time() - start_time
    print(
        "[labels] inserted=%d considered=%d skipped_mid=%d skipped_future=%d elapsed=%.2fs"
        % (total_inserted, total_considered, total_skipped_mid, total_skipped_future, elapsed)
    )
    conn.close()


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build labels_outcome from replay DB.")
    parser.add_argument("-DatasetId", required=True)
    parser.add_argument("-Horizons", default="60,120")
    parser.add_argument("-BpMode", default="mid")
    parser.add_argument("-Thresholds", default="+8,-6")
    return parser


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()
    run_job(args)


if __name__ == "__main__":
    main()
