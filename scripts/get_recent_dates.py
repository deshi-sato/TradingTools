# prints "YYYY-MM-DD,YYYY-MM-DD" (oldest,newest) for the most recent two trading days
import sqlite3, sys
DB = r"data/rss_daily.db"
with sqlite3.connect(DB) as c:
    rows = c.execute("select distinct date from daily_bars order by date desc limit 2").fetchall()
dates = [r[0] for r in rows][::-1]
if len(dates) < 2:
    print("")
    sys.exit(1)
print(",".join(dates))
