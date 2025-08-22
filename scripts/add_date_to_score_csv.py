# Add a "date" column to data/score_daily.csv if missing, using MAX(date) from DB
import sqlite3, pandas as pd, os, sys
DB  = r"data/rss_daily.db"
CSV = r"data/score_daily.csv"
if not os.path.exists(CSV):
    print("[ERROR] not found:", CSV); sys.exit(1)
with sqlite3.connect(DB) as c:
    latest = c.execute("select max(date) from daily_bars").fetchone()[0]
df = pd.read_csv(CSV)
if "date" not in df.columns:
    df.insert(0, "date", latest)
df.to_csv(CSV, index=False, encoding="utf-8")
print("patched with date:", latest, "rows:", len(df))
