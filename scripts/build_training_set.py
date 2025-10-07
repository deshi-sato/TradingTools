#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_training_set.py

Exports a vertical training CSV by joining labels_outcome with features_stream.
Uses dataset_registry to locate the correct dated refeed DB.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

JST_DEFAULT_COLS = [
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
]


def parse_cols(arg: Optional[str]) -> List[str]:
    if not arg:
        return list(JST_DEFAULT_COLS)
    cols = [piece.strip() for piece in arg.split(",") if piece.strip()]
    if not cols:
        return list(JST_DEFAULT_COLS)
    return cols


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


def ensure_indexes(conn: sqlite3.Connection) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_labels_dataset_symbol_ts ON labels_outcome(dataset_id, symbol, ts)",
        "CREATE INDEX IF NOT EXISTS idx_features_symbol_ts ON features_stream(symbol, t_exec)",
    ]
    for stmt in statements:
        conn.execute(stmt)
    conn.commit()


def fetch_rows(
    conn: sqlite3.Connection,
    dataset_id: str,
    extra_cols: Sequence[str],
) -> Iterable[Tuple]:
    select_cols = ", ".join(
        [
            "l.symbol",
            "l.ts",
            "l.horizon_sec",
            "l.ret_bp",
            "l.label",
        ]
        + [f"f.{col}" for col in extra_cols]
    )
    join_conditions = ["f.symbol = l.symbol", "CAST(f.t_exec * 1000 AS INTEGER) = l.ts"]
    sql = f"""
        SELECT {select_cols}
          FROM labels_outcome AS l
          JOIN features_stream AS f
            ON {' AND '.join(join_conditions)}
         WHERE l.dataset_id=?
         ORDER BY l.symbol, l.ts, l.horizon_sec
    """
    return conn.execute(sql, (dataset_id,))


def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(
    out_path: Path,
    rows: Iterable[Tuple],
    headers: Sequence[str],
) -> int:
    ensure_output_path(out_path)
    # use utf-8-sig to emit BOM
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\r\n")
        writer.writerow(headers)
        count = 0
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def run(args: argparse.Namespace) -> None:
    dataset_id = args.DatasetId
    extra_cols = parse_cols(args.Cols)
    out_path = Path(args.Out) if args.Out else Path(f"exports/trainset_{dataset_id}.csv")

    db_path = resolve_db_path(dataset_id)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_indexes(conn)

    headers = ["symbol", "ts", "horizon_sec", "ret_bp", "label"] + list(extra_cols)
    rows = fetch_rows(conn, dataset_id, extra_cols)
    count = write_csv(out_path, rows, headers)

    conn.close()
    print(f"[training_set] rows={count} out={out_path}")


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build training CSV from labels_outcome.")
    parser.add_argument("-DatasetId", required=True)
    parser.add_argument("-Out")
    parser.add_argument("-Cols")
    return parser


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
