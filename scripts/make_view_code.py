\"\"\"make_view_code.py : Auto-generated placeholder

- file: scripts/make_view_code.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
import sqlite3, os

DB = r"data/rss_daily.db"
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(DB)
conn.executescript("""
CREATE VIEW IF NOT EXISTS v_daily_bars AS
SELECT
  date,
  ticker,
  REPLACE(ticker, '.T','') AS code,
  open, high, low, close, adj_close, volume
FROM daily_bars;
""")
conn.commit(); conn.close()
print("View v_daily_bars created.")
