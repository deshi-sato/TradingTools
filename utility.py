\"\"\"utility.py : Auto-generated placeholder

- file: utility.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
from datetime import datetime, timedelta
import pandas as pd

# 日付列（例: 2025-08-08）と時刻列（例: 09:15:00）結合して
# 「2025-08-08 09:15:00」という1つの時系列キーを作成する関数
def parse_date_time(row_date, row_time):
    if isinstance(row_date, str):
        row_date = pd.to_datetime(row_date).date()
    elif isinstance(row_date, datetime):
        row_date = row_date.date()
    if isinstance(row_time, str):
        row_time = pd.to_datetime(row_time).time()
    elif isinstance(row_time, datetime):
        row_time = row_time.time()
    return datetime.combine(row_date, row_time)

def filter_top(df, min_count=5):
    grouped = df.groupby("合計スコア").size().sort_index(ascending=False)
    total = 0
    threshold = 0
    for score, count in grouped.items():
        total += count
        threshold = score
        if total >= min_count:
            break
    return df[df["合計スコア"] >= threshold]

def get_japan_market_today():
    now = datetime.now()
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now < market_start:
        # 9:00より前 → 前日を「今日」とする
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # 9:00以降 → 通常の今日
        return now.strftime("%Y-%m-%d")

