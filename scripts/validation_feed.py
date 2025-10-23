#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
validation_feed.py

Replay features_stream and raw_push rows from a refeed database into a fresh
validation database while rebasing timestamps to "now". The tool is intended
to drive paper-mode runners against historical orderbook ticks as if they were live.

Usage (PowerShell):

    py -m scripts.validation_feed `
        --source db\naut_market_20251020_refeed.db `
        --dest data\naut_market_validation.db `
        --dataset-id VALIDATION_FEED `
        --speed 2.0
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

DEFAULT_DATASET_ID = "VALIDATION_FEED"
REFEED_GLOB = "naut_market_*_refeed.db"
DB_ROOT = Path("db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_push(
  t_recv REAL NOT NULL,
  topic TEXT,
  symbol TEXT,
  payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS features_stream(
  dataset_id   TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  t_exec       REAL NOT NULL,
  ts_ms        INTEGER NOT NULL,
  ver          TEXT NOT NULL,
  f1           REAL,
  f2           REAL,
  f3           REAL,
  f4           REAL,
  f5           REAL,
  f6           REAL,
  score        REAL,
  spread_ticks INTEGER,
  bid1         REAL,
  ask1         REAL,
  bidqty1      REAL,
  askqty1      REAL,
  v_spike      REAL DEFAULT 0,
  v_rate       REAL DEFAULT 0,
  f1_delta     REAL DEFAULT 0
);
"""

INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_features_stream_symbol_ts ON features_stream(symbol, t_exec);",
    "CREATE INDEX IF NOT EXISTS idx_raw_push_ts ON raw_push(t_recv);",
    "CREATE INDEX IF NOT EXISTS idx_raw_push_symbol_ts ON raw_push(symbol, t_recv);",
)


def resolve_latest_refeed(db_root: Path = DB_ROOT) -> Path:
    candidates = sorted(db_root.glob(REFEED_GLOB), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No refeed database found under {db_root} matching {REFEED_GLOB}")
    return candidates[0]


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    for statement in INDEX_SQL:
        conn.execute(statement)
    conn.commit()


def iter_features(conn: sqlite3.Connection) -> Iterator[Dict[str, float]]:
    query = """
        SELECT dataset_id, symbol, t_exec, ts_ms, ver,
               f1, f2, f3, f4, f5, f6,
               score, spread_ticks, bid1, ask1,
               bidqty1, askqty1, v_spike, v_rate, f1_delta
        FROM features_stream
        ORDER BY t_exec ASC
    """
    conn.row_factory = sqlite3.Row
    cur = conn.execute(query)
    for row in cur:
        yield dict(row)


def iter_raw_push(conn: sqlite3.Connection) -> Iterator[Dict[str, Any]]:
    query = """
        SELECT t_recv, topic, symbol, payload
        FROM raw_push
        ORDER BY t_recv ASC
    """
    conn.row_factory = sqlite3.Row
    cur = conn.execute(query)
    for row in cur:
        yield dict(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay refeed DB into validation DB with rebased timestamps.")
    parser.add_argument("--source", help="Source refeed DB (default: latest db/naut_market_YYYYMMDD_refeed.db).")
    parser.add_argument(
        "--dest",
        default="naut_market.db",
        help="Destination validation DB path (default: ./naut_market.db). Existing file is recreated.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Dataset ID to stamp in the validation DB (default: %(default)s).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 1.0 preserves original spacing; 2.0 doubles speed; "
        "values <=0 disable sleeping.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="Optional seconds to wait before injecting the first tick (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of rows to replay (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    source_path = Path(args.source) if args.source else resolve_latest_refeed()
    if not source_path.exists():
        raise FileNotFoundError(f"Source DB not found: {source_path}")

    dest_path = Path(args.dest)
    if dest_path.exists():
        logging.info("Removing existing validation DB: %s", dest_path)
        dest_path.unlink()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("validation dest=%s source=%s dataset_id=%s speed=%.2f", dest_path, source_path, args.dataset_id, args.speed)

    src_conn = sqlite3.connect(source_path, check_same_thread=False)
    dest_conn = sqlite3.connect(dest_path, check_same_thread=False)
    dest_conn.execute("PRAGMA journal_mode=WAL;")
    dest_conn.execute("PRAGMA synchronous=NORMAL;")
    ensure_schema(dest_conn)

    feature_iter = iter_features(src_conn)
    raw_iter = iter_raw_push(src_conn)
    next_feature = next(feature_iter, None)
    next_raw = next(raw_iter, None)

    seen_pairs: set[Tuple[str, float]] = set()

    max_rows = args.limit if args.limit and args.limit > 0 else None
    features_streamed = 0
    raw_streamed = 0
    baseline_ts: Optional[float] = None
    start_wall = time.time() + max(0.0, args.start_delay)
    speed = max(0.0, args.speed)

    if args.start_delay > 0:
        logging.info("Initial delay %.2fs before streaming.", args.start_delay)
        time.sleep(args.start_delay)

    while next_feature is not None or next_raw is not None:
        use_feature = False
        feature_ts: Optional[float] = None
        raw_ts: Optional[float] = None

        if next_feature is not None:
            feature_ts = float(next_feature.get("t_exec") or 0.0)
        if next_raw is not None:
            raw_ts = float(next_raw.get("t_recv") or 0.0)

        if next_feature is not None and (next_raw is None or (feature_ts or 0.0) <= (raw_ts or float("inf"))):
            row = next_feature
            original_ts = feature_ts or 0.0
            use_feature = True
            symbol = str(row.get("symbol") or "")
            if not symbol:
                next_feature = next(feature_iter, None)
                continue
            key = (symbol, original_ts)
            if key in seen_pairs:
                next_feature = next(feature_iter, None)
                continue
        else:
            row = next_raw
            original_ts = raw_ts or 0.0

        if row is None:
            break

        if baseline_ts is None:
            baseline_ts = original_ts

        offset = max(0.0, original_ts - baseline_ts)

        if speed > 0:
            target_wall = start_wall + offset / speed
            now = time.time()
            sleep_for = target_wall - now
            if sleep_for > 0:
                time.sleep(sleep_for)
        else:
            target_wall = time.time()

        if use_feature:
            new_t_exec = target_wall
            ts_ms = int(round(new_t_exec * 1000))
            dest_conn.execute(
                """
                INSERT INTO features_stream(
                    dataset_id, symbol, t_exec, ts_ms, ver, f1, f2, f3, f4, f5, f6,
                    score, spread_ticks, bid1, ask1, bidqty1, askqty1, v_spike, v_rate, f1_delta
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    args.dataset_id,
                    row.get("symbol"),
                    new_t_exec,
                    ts_ms,
                    row.get("ver"),
                    row.get("f1"),
                    row.get("f2"),
                    row.get("f3"),
                    row.get("f4"),
                    row.get("f5"),
                    row.get("f6"),
                    row.get("score"),
                    row.get("spread_ticks"),
                    row.get("bid1"),
                    row.get("ask1"),
                    row.get("bidqty1"),
                    row.get("askqty1"),
                    row.get("v_spike"),
                    row.get("v_rate"),
                    row.get("f1_delta"),
                ),
            )
            dest_conn.commit()
            features_streamed += 1
            seen_pairs.add(key)
            next_feature = next(feature_iter, None)
            if max_rows and features_streamed >= max_rows:
                break
        else:
            dest_conn.execute(
                """
                INSERT INTO raw_push(t_recv, topic, symbol, payload)
                VALUES(?,?,?,?)
                """,
                (
                    target_wall,
                    row.get("topic"),
                    row.get("symbol"),
                    row.get("payload"),
                ),
            )
            dest_conn.commit()
            raw_streamed += 1
            next_raw = next(raw_iter, None)

    logging.info(
        "Replay complete features=%d raw_push=%d dest=%s",
        features_streamed,
        raw_streamed,
        dest_path,
    )
    src_conn.close()
    dest_conn.close()


if __name__ == "__main__":
    main()
