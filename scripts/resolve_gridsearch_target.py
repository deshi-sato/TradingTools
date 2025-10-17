#!/usr/bin/env python
"""
scripts/resolve_gridsearch_target.py

目的:
  最新の refeed DB とそれに紐づく dataset_id を推定し、
  PowerShell で使える `$ds` / `$refeed` の設定行を出力する。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_ROOT = Path("db")


def iter_refeed_paths() -> list[Path]:
    return sorted(DB_ROOT.glob("naut_market_*_refeed.db"), reverse=True)


def resolve_dataset(refeed: Path) -> tuple[str, str] | None:
    with sqlite3.connect(str(refeed)) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT dataset_id
              FROM dataset_registry
          ORDER BY updated_at DESC
             LIMIT 1
            """
        ).fetchone()
        if row and row["dataset_id"]:
            return row["dataset_id"], str(refeed)

        row = conn.execute(
            "SELECT dataset_id FROM features_stream ORDER BY ts_ms DESC LIMIT 1"
        ).fetchone()
        if row and row["dataset_id"]:
            return row["dataset_id"], str(refeed)

    return None


def main() -> None:
    for candidate in iter_refeed_paths():
        result = resolve_dataset(candidate)
        if result:
            dataset_id, path_str = result
            # PowerShell でそのまま貼れるように出力する
            print(f"$ds      = '{dataset_id}'")
            print(f"$refeed  = '{path_str.replace('/', '\\\\')}'")
            return
    raise SystemExit("ERROR: 有効な dataset_id が見つかりませんでした。")


if __name__ == "__main__":
    main()
