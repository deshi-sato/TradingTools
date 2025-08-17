"""
Delete today's minute_data rows at/after 11:30 (local time).

Usage:
  python delete_after_1130.py                 # dry-run (no deletion)
  python delete_after_1130.py --execute       # actually delete
  python delete_after_1130.py --db path/to.db # specify DB

Notes:
  - Targets table `minute_data` with columns including `datetime` and `ticker`.
  - Matches rows where date(datetime) == date('now','localtime')
    AND time(datetime) >= '11:30:00'.
  - Uses SQLite date/time functions; assumes stored datetimes are local-like
    strings (consistent with existing queries in this repo).
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delete today's data at/after 11:30 from minute_data")
    p.add_argument("--db", default=str(Path(__file__).resolve().parent / "data" / "rss_data.db"),
                   help="Path to rss_data.db (default: data/rss_data.db)")
    p.add_argument("--execute", action="store_true", help="Actually delete rows (otherwise dry-run)")
    return p.parse_args()


def main() -> None:
    args = build_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Count candidates first
    count_sql = (
        "SELECT COUNT(*) FROM minute_data "
        "WHERE date(datetime) = date('now','localtime') "
        "AND time(datetime) >= '11:30:00'"
    )
    (cnt,) = cur.execute(count_sql).fetchone()

    # Show a small preview for verification
    preview_sql = (
        "SELECT ticker, datetime FROM minute_data "
        "WHERE date(datetime) = date('now','localtime') "
        "AND time(datetime) >= '11:30:00' "
        "ORDER BY datetime ASC LIMIT 5"
    )
    preview = cur.execute(preview_sql).fetchall()

    print(f"DB: {db_path}")
    print(f"Rows to delete (today >= 11:30): {cnt}")
    if preview:
        print("Sample (first 5):")
        for t, dt in preview:
            print(f"  {dt}  {t}")

    if not args.execute:
        print("Dry-run complete. Use --execute to perform deletion.")
        conn.close()
        return

    # Perform deletion
    delete_sql = (
        "DELETE FROM minute_data "
        "WHERE date(datetime) = date('now','localtime') "
        "AND time(datetime) >= '11:30:00'"
    )
    cur.execute(delete_sql)
    conn.commit()
    print(f"Deleted rows: {cnt}")
    conn.close()


if __name__ == "__main__":
    main()

