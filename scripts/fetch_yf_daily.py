#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOPIX100 の銘柄コード（例: 7203）を data/topix100_codes.txt から読み込み、
Yahoo Finance の日足（1d）を取得して SQLite: data/rss_daily.db に保存する。

仕様:
- テーブル: daily_bars (date, ticker, open, high, low, close, adj_close, volume)
- PRIMARY KEY (ticker, date)
- 期間: 日本の前営業日を終点として 1 年分
- 取引所サフィックスは自動で .T を付与（yfinance 形式）
- 冪等実行可能（INSERT OR REPLACE）
- 祝日判定には pandas_market_calendars(=XTKS) を使用、不可なら平日フォールバック
"""

import os
import sys
import time
import argparse
import sqlite3
import datetime as dt
from typing import List, Optional

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---- 祝日/営業日計算（XTKS / fallback）--------------------------------------


def last_trading_day_japan(end_hint: Optional[dt.date] = None) -> dt.date:
    """
    日本の前営業日（東京証券取引所）を返す。
    pandas_market_calendars が利用可能なら XTKS カレンダーに準拠。
    使えない場合は「直近日の平日（Mon-Fri）」にフォールバック。
    """
    if end_hint is None:
        end_hint = dt.date.today()

    # まずは XTKS で厳密に
    try:
        import pandas_market_calendars as pmc
        xtks = pmc.get_calendar("XTKS")
        # 'end_hint' の前日までで直近のセッション日を探す
        # 例：今日が 2025-08-19（火）なら、前営業日は 2025-08-18（月）
        sched = xtks.schedule(start_date=(end_hint - dt.timedelta(days=14)), end_date=end_hint)
        # schedule の index はナイーブな日付（UTC）相当。営業日終了が含まれているので前日営業日を取る
        # 当日が営業日でも「前営業日」が欲しいため、end_hint 当日の行を除外して最後を取る
        dates = pd.to_datetime(sched.index).date
        prevs = [d for d in dates if d < end_hint]
        if not prevs:
            # 直近 2 週間に前営業日が無いケースは実質ありえないが、一応フォールバック
            raise RuntimeError("No previous trading day found in XTKS schedule window.")
        return prevs[-1]
    except Exception:
        # フォールバック：平日ベース
        d = end_hint - dt.timedelta(days=1)
        # 月〜金まで下がる
        while d.weekday() >= 5:  # 5=Sat, 6=Sun
            d -= dt.timedelta(days=1)
        return d


def one_year_ago(d: dt.date) -> dt.date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:
        # うるう年対応（2/29 → 2/28）
        return d - dt.timedelta(days=365)


# ---- DB スキーマ -------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS daily_bars (
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL,
  adj_close REAL, volume INTEGER,
  PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date);
CREATE INDEX IF NOT EXISTS idx_daily_bars_ticker ON daily_bars(ticker);
"""


# ---- 引数 --------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--codes_file", default="data/topix100_codes.txt",
                   help="1行1銘柄コード（例: 7203）。.T は自動付与")
    p.add_argument("--db", default="data/rss_daily.db")
    p.add_argument("--sleep", type=float, default=0.6, help="リクエスト間隔（秒）")
    p.add_argument("--auto_adjust", action="store_true",
                   help="分割/配当調整後の価格を close に反映（adj_close も保存）")
    return p.parse_args()


def load_tickers(codes_file: str) -> List[str]:
    if not os.path.exists(codes_file):
        print(f"[ERROR] codes_file not found: {codes_file}", file=sys.stderr)
        sys.exit(1)
    codes = [x.strip() for x in open(codes_file, "r", encoding="utf-8").read().splitlines() if x.strip()]
    # yfinance 用に .T を付与
    tickers = [f"{c}.T" if not c.endswith(".T") else c for c in codes]
    return sorted(set(tickers))


def ensure_schema(conn: sqlite3.Connection):
    for stmt in DDL.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s + ";")
    conn.commit()


# ---- 取得と保存 --------------------------------------------------------------


@retry(reraise=True, stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=1, max=16),
       retry=retry_if_exception_type(Exception))
def fetch_one(ticker: str, start: str, end: str, auto_adjust: bool) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end,
        interval="1d", auto_adjust=auto_adjust, progress=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"no data for {ticker} in {start}..{end}")
    df = df.reset_index().rename(columns=str.lower)
    # yfinance 列名整形
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["date","ticker","open","high","low","close","adj_close","volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def upsert(conn: sqlite3.Connection, df: pd.DataFrame):
    rows = [tuple(x) for x in df.itertuples(index=False, name=None)]
    conn.executemany("""
        INSERT OR REPLACE INTO daily_bars
        (date,ticker,open,high,low,close,adj_close,volume)
        VALUES (?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()


# ---- メイン ------------------------------------------------------------------


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.db), exist_ok=True)

    # 期間の決定：前営業日を end、そこから 1 年前を start
    end_date = last_trading_day_japan(dt.date.today())
    start_date = one_year_ago(end_date)

    tickers = load_tickers(args.codes_file)
    conn = sqlite3.connect(args.db)
    ensure_schema(conn)

    print(f"Target window: {start_date} → {end_date}  (inclusive)")
    print(f"Tickers: {len(tickers)} symbols")

    # yfinance の end は「非包含」なので、end_date の翌日を渡す
    yf_end_exclusive = (end_date + dt.timedelta(days=1)).isoformat()
    yf_start = start_date.isoformat()

    for i, tkr in enumerate(tickers, 1):
        try:
            df = fetch_one(tkr, yf_start, yf_end_exclusive, args.auto_adjust)
            upsert(conn, df)
            print(f"[{i}/{len(tickers)}] {tkr} {df['date'].min()} → {df['date'].max()}  {len(df)} rows")
            time.sleep(args.sleep)
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {tkr} ERROR: {e}", file=sys.stderr)

    conn.close()
    print(f"✅ Done. DB: {args.db}")


if __name__ == "__main__":
    main()

