#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOPIX100 の日足を yfinance から増分取得し、SQLite に UPSERT します。

要件（codex_trend_score.md より要約）:
- 対象は DB の最終日以降のみ（新規ティッカーは lookback 日）
- 当日分は除外（前日まで、JST基準）
- 失敗は銘柄単位で握りつぶし継続。再実行は冪等（UPSERT）
- 既存の一括取得ではなく、増分更新が標準動作
- DB: C:\\Users\\Owner\\Documents\\desshi_signal_viewer\\rss_daily.db（既定）
- テーブル: daily_bars(ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY(ticker,date))
- CLI: --db, --tickers-file, --lookback-days, --max-workers, --since, --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


# ---- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch TOPIX100 daily bars incrementally and upsert into SQLite.")
    p.add_argument(
        "--db",
        default=r"C:\\Users\\Owner\\Documents\\desshi_signal_viewer\\rss_daily.db",
        help="Path to SQLite DB (default: repository rss_daily.db)",
    )
    p.add_argument(
        "--tickers-file",
        dest="tickers_file",
        default=None,
        help="Tickers list file (CSV/TSV/newline, 1st column). If omitted, fall back to data/topix100_codes.txt",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=740,
        help="Initial backfill days for unseen tickers (default 740 for ~2 years)",
    )
    p.add_argument("--max-workers", type=int, default=6, help="Concurrent fetch workers (default 6)")
    p.add_argument("--since", type=str, default=None, help="Force start date YYYY-MM-DD (overrides DB max)")
    p.add_argument("--dry-run", action="store_true", help="Fetch and compute only; do not write DB")
    return p.parse_args()


# ---- Utilities ----------------------------------------------------------------


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars(
          ticker TEXT NOT NULL,
          date   TEXT NOT NULL,
          open   REAL, high REAL, low REAL, close REAL, volume INTEGER,
          PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.commit()


def _normalize_ticker(code: str) -> str:
    code = code.strip()
    if not code or code.startswith("#"):
        return ""
    # 1列目を採用し、カンマ/タブで区切られていれば先頭だけ使う
    if "," in code:
        code = code.split(",", 1)[0]
    if "\t" in code:
        code = code.split("\t", 1)[0]
    code = code.strip()
    if not code:
        return ""
    return code if code.endswith(".T") else f"{code}.T"


def load_tickers_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            t = _normalize_ticker(raw)
            if t:
                tickers.append(t)
    # 重複除去し安定ソート
    return sorted(set(tickers))


def load_default_topix100() -> List[str]:
    # 既存のリスト（従来コード互換）
    default_path = os.path.join("data", "topix100_codes.txt")
    if not os.path.exists(default_path):
        logging.warning("Default tickers file not found: %s", default_path)
        return []
    return load_tickers_from_file(default_path)


def jst_today() -> date:
    return datetime.now(JST).date()


def compute_end_date_exclusive() -> Tuple[date, date]:
    """
    Returns (end_inclusive, end_exclusive) where inclusive is yesterday in JST.
    """
    today = jst_today()
    end_inclusive = today - timedelta(days=1)
    return end_inclusive, end_inclusive + timedelta(days=1)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def get_db_max_date(conn: sqlite3.Connection, ticker: str) -> Optional[date]:
    cur = conn.execute("SELECT MAX(date) FROM daily_bars WHERE ticker=?", (ticker,))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    try:
        return parse_date(row[0])
    except Exception:
        return None


def history_df(ticker: str, start_date: date, end_exclusive: date) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(
        start=str(start_date), end=str(end_exclusive), interval="1d", auto_adjust=False
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])  # empty
    # 標準化
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    # index から JST 基準の日付列を作成
    idx = df.index
    try:
        # tz-aware の場合は JST に変換
        if getattr(idx, "tz", None) is not None:
            dates = idx.tz_convert(JST).date
        else:
            dates = pd.to_datetime(idx).date
    except Exception:
        dates = pd.to_datetime(idx).date
    df["date"] = dates
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna(how="any")
    return df


def to_rows(ticker: str, df: pd.DataFrame) -> List[Tuple[str, str, float, float, float, float, int]]:
    rows: List[Tuple[str, str, float, float, float, float, int]] = []
    for d, o, h, l, c, v in df.itertuples(index=False, name=None):
        ds = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        rows.append((ticker, ds, float(o), float(h), float(l), float(c), int(v or 0)))
    return rows


def upsert_rows(conn: sqlite3.Connection, rows: Sequence[Tuple[str, str, float, float, float, float, int]]) -> int:
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO daily_bars(ticker,date,open,high,low,close,volume)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(ticker,date) DO UPDATE SET
          open=excluded.open,
          high=excluded.high,
          low=excluded.low,
          close=excluded.close,
          volume=excluded.volume
        """,
        rows,
    )
    return len(rows)


@dataclass
class Task:
    ticker: str
    since: date
    end_inclusive: date


def worker(db_path: str, task: Task, dry_run: bool) -> Tuple[str, int, Optional[str]]:
    """Fetch and upsert one ticker. Returns (ticker, upserted_count, error_message)."""
    t = task.ticker
    start = task.since
    end_incl = task.end_inclusive
    logging.info("%s start=%s end=%s", t, start, end_incl)
    try:
        df = history_df(t, start, end_incl + timedelta(days=1))
        if df.empty:
            logging.info("%s no rows", t)
            return t, 0, None
        rows = to_rows(t, df)
        if dry_run:
            logging.info("%s rows=%d (dry-run)", t, len(rows))
            return t, 0, None
        conn = sqlite3.connect(db_path, timeout=30, isolation_level=None, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            ensure_schema(conn)
            conn.execute("BEGIN")
            n = upsert_rows(conn, rows)
            conn.commit()
        finally:
            conn.close()
        logging.info("%s rows=%d", t, n)
        return t, n, None
    except Exception as e:
        msg = str(e)
        logging.warning("%s error: %s", t, msg)
        # 軽いバックオフ（429 等）
        if "429" in msg:
            try:
                import time as _time

                _time.sleep(1.0)
            except Exception:
                pass
        return t, 0, msg


def main() -> None:
    args = parse_args()

    # ロガー
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    # DB 準備
    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = sqlite3.connect(args.db, timeout=30)
    try:
        ensure_schema(conn)
    finally:
        conn.close()
    logging.info("Using DB: %s", args.db)

    # ティッカー集合
    if args.tickers_file:
        tickers = load_tickers_from_file(args.tickers_file)
    else:
        tickers = load_default_topix100()
    if not tickers:
        logging.error("No tickers resolved. Provide --tickers-file or prepare data/topix100_codes.txt")
        sys.exit(1)

    # 期間
    end_inclusive, end_exclusive = compute_end_date_exclusive()
    logging.info("End date (inclusive, JST): %s", end_inclusive)

    forced_since: Optional[date] = None
    if args.since:
        try:
            forced_since = parse_date(args.since)
        except Exception:
            logging.error("Invalid --since format. Use YYYY-MM-DD")
            sys.exit(2)

    # 各ティッカーの since を決定
    tasks: List[Task] = []
    conn = sqlite3.connect(args.db, timeout=30)
    try:
        for t in tickers:
            maxd = get_db_max_date(conn, t)
            if maxd is None:
                since = end_inclusive - timedelta(days=int(args.lookback_days))
            else:
                since = maxd + timedelta(days=1)
            if forced_since is not None:
                # Allow backfilling earlier than DB max by honoring the earlier date
                since = min(since, forced_since)
            if since > end_inclusive:
                logging.info("%s up-to-date (since=%s > end=%s)", t, since, end_inclusive)
                continue
            tasks.append(Task(ticker=t, since=since, end_inclusive=end_inclusive))
    finally:
        conn.close()

    if not tasks:
        logging.info("All tickers are up-to-date. Nothing to do.")
        return

    # 並列処理
    total_upserts = 0
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        futs = [ex.submit(worker, args.db, task, args.dry_run) for task in tasks]
        for fut in as_completed(futs):
            t, n, err = fut.result()
            if not err:
                total_upserts += n

    logging.info("Total upserts: %d", total_upserts)


if __name__ == "__main__":
    main()
