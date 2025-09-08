#!/usr/bin/env python3
"""
Export minute bars from sqlite3 to per-symbol CSV files.

Requirements from spec:
- Read from table `minute_data(ticker, datetime, open, high, low, close, volume)`
- Args: --db, --out, --start, --end, --symbols (comma-separated, optional)
- Output: out/{symbol}.csv with columns: timestamp,open,high,low,close,volume,symbol
- Timestamp must be ISO8601, sorted by symbol,timestamp
- Friendly error messages
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export minute_data records from an SQLite database to per-symbol CSV files."
        )
    )
    p.add_argument(
        "--db",
        required=True,
        help="Path to SQLite database file containing table minute_data",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory to write CSV files (created if missing)",
    )
    p.add_argument(
        "--start",
        required=True,
        help="Start datetime (inclusive). e.g. '2025-01-01 00:00:00' or ISO-8601",
    )
    p.add_argument(
        "--end",
        required=True,
        help="End datetime (exclusive or inclusive; stored as <=). e.g. '2025-09-01 00:00:00'",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Optional comma-separated list of tickers to filter (e.g. '7011,5803')",
    )
    return p.parse_args()


def friendly_exit(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_iso8601(ts: object) -> str:
    """Best-effort conversion to ISO8601 string with seconds precision.

    Accepts str (common formats), int/float (epoch seconds), or datetime.
    Falls back to original string if parsing fails.
    """
    if ts is None:
        return ""
    if isinstance(ts, datetime):
        # Normalize to seconds, preserve tz if present
        return ts.isoformat(timespec="seconds")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(
                timespec="seconds"
            )
        except Exception:
            return str(ts)
    s = str(ts).strip()
    if not s:
        return s
    # Replace space with 'T' to satisfy ISO8601 if the format is YYYY-MM-DD HH:MM:SS
    # Try Python's permissive ISO parser first
    try:
        iso = s.replace("Z", "+00:00")
        return datetime.fromisoformat(iso).isoformat(timespec="seconds")
    except Exception:
        pass
    # Try common patterns
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y%m%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.isoformat(timespec="seconds")
        except Exception:
            continue
    # Fallback: return as-is; caller sorted order will still use DB ordering
    return s.replace(" ", "T")


def open_db(path: str) -> sqlite3.Connection:
    if not os.path.isfile(path):
        friendly_exit(
            f"Database file not found: {path}\n"
            "- Check the --db path. If it contains spaces, quote it.\n"
            "- Example: --db \"C:\\path\\to\\rss_data.db\"",
            code=2,
        )
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error as e:
        friendly_exit(f"Failed to open database: {e}")
    conn.row_factory = sqlite3.Row
    return conn


def build_query(
    symbols: Optional[List[str]],
    start: str,
    end: str,
) -> Tuple[str, List[object]]:
    sql = (
        "SELECT ticker, datetime, open, high, low, close, volume "
        "FROM minute_data WHERE 1=1"
    )
    params: List[object] = []
    if symbols:
        placeholders = ",".join(["?"] * len(symbols))
        sql += f" AND ticker IN ({placeholders})"
        params.extend(symbols)
    # Use BETWEEN inclusive bounds; end inclusive by requirement ambiguity; choose <= for end
    sql += " AND datetime >= ? AND datetime <= ?"
    params.extend([start, end])
    # Sort for streaming grouped write
    sql += " ORDER BY ticker ASC, datetime ASC"
    return sql, params


def export_rows(conn: sqlite3.Connection, out_dir: str, sql: str, params: List[object]) -> int:
    os.makedirs(out_dir, exist_ok=True)

    cur = conn.cursor()
    try:
        cur.execute(sql, params)
    except sqlite3.OperationalError as e:
        # Likely missing table/columns
        friendly_exit(
            f"SQL error: {e}\n"
            "- Ensure table 'minute_data' exists with columns: "
            "ticker, datetime, open, high, low, close, volume",
            code=3,
        )

    writers: Dict[str, Tuple[csv.writer, any]] = {}
    written = 0
    try:
        for row in cur:
            symbol = str(row["ticker"]) if row["ticker"] is not None else ""
            ts = ensure_iso8601(row["datetime"])  # type: ignore[index]
            out_path = os.path.join(out_dir, f"{symbol}.csv")

            if symbol not in writers:
                f = open(out_path, "w", newline="", encoding="utf-8")
                w = csv.writer(f)
                w.writerow(["timestamp", "open", "high", "low", "close", "volume", "symbol"])
                writers[symbol] = (w, f)

            w, _f = writers[symbol]
            w.writerow(
                [
                    ts,
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    symbol,
                ]
            )
            written += 1
    finally:
        # Close all open files
        for w, f in writers.values():
            try:
                f.flush()
            except Exception:
                pass
            f.close()
        cur.close()

    return written


def main() -> None:
    args = parse_args()

    # Normalize and parse symbols
    symbols: Optional[List[str]] = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if not symbols:
            symbols = None

    # Basic sanity checks for date range (not strict; DB performs actual filter)
    start = args.start
    end = args.end
    if not start or not end:
        friendly_exit("Both --start and --end must be provided.")

    conn = open_db(args.db)
    try:
        sql, params = build_query(symbols, start, end)
        total = export_rows(conn, args.out, sql, params)
    finally:
        conn.close()

    if total == 0:
        friendly_exit(
            "No rows matched the given filters (symbols/date range).\n"
            "- Check that the date range matches the data.\n"
            "- If you specified --symbols, confirm tickers exist in the DB.",
            code=4,
        )

    print(
        f"Export completed: {total} rows written to per-symbol CSVs in '{args.out}'."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        friendly_exit("Interrupted by user.", code=130)

