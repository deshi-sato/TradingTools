#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ensure_registry_schema.py

ユーティリティ: dataset_registry テーブルのスキーマを統一し、欠損している列を補完する。

使い方:
    py scripts/ensure_registry_schema.py db/naut_market_20251014_refeed.db [他のDB...]
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Iterable

UNIFIED_DDL = """
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
)
"""


def _compute_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(dataset_registry)")}
    for column, col_type in [
        ("db_path", "TEXT"),
        ("source_db_path", "TEXT"),
        ("build_tool", "TEXT"),
        ("code_version", "TEXT"),
        ("config_json", "TEXT"),
        ("db_sha1", "TEXT"),
        ("source_db_sha1", "TEXT"),
        ("regime_tag", "TEXT"),
        ("created_at", "TEXT"),
        ("updated_at", "TEXT"),
    ]:
        if column not in existing:
            conn.execute(f"ALTER TABLE dataset_registry ADD COLUMN {column} {col_type}")


def ensure_registry_schema(db_path: Path) -> None:
    db_path = db_path.resolve()
    if not db_path.exists():
        raise SystemExit(f"[ERROR] DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(UNIFIED_DDL)
        _ensure_columns(conn)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        sha1 = _compute_sha1(db_path)
        abs_path = str(db_path)

        rows = conn.execute("SELECT * FROM dataset_registry").fetchall()
        for row in rows:
            dataset_id = row["dataset_id"]
            if dataset_id is None:
                continue
            updates = {}
            if not row["db_path"]:
                updates["db_path"] = abs_path
            if not row["source_db_path"]:
                updates["source_db_path"] = abs_path
            if not row["db_sha1"]:
                updates["db_sha1"] = sha1
            if not row["source_db_sha1"]:
                updates["source_db_sha1"] = sha1
            if not row["created_at"]:
                updates["created_at"] = now
            updates["updated_at"] = now
            if not row["config_json"]:
                updates["config_json"] = "{}"

            if updates:
                set_clause = ", ".join(f"{col}=?" for col in updates.keys())
                values = list(updates.values()) + [dataset_id]
                conn.execute(
                    f"UPDATE dataset_registry SET {set_clause} WHERE dataset_id=?",
                    values,
                )

        conn.commit()
        print(f"[OK] dataset_registry schema unified -> {db_path}")
    finally:
        conn.close()


def _iter_db_targets(args: argparse.Namespace) -> Iterable[Path]:
    if args.dbs:
        for item in args.dbs:
            yield Path(item)
    else:
        db_root = Path("db")
        for candidate in sorted(db_root.glob("*.db")):
            yield candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify dataset_registry schema")
    parser.add_argument("dbs", nargs="*", help="Target SQLite DB paths")
    args = parser.parse_args()

    for target in _iter_db_targets(args):
        ensure_registry_schema(target)


if __name__ == "__main__":
    main()
