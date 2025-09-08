\"\"\"probe_completed.py : Auto-generated placeholder

- file: probe_completed.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
import sqlite3
from pathlib import Path

DB = Path('data')/'rss_data.db'
conn = sqlite3.connect(str(DB))
cur = conn.cursor()

rows = cur.execute(
    """
    SELECT ticker, date(datetime) as d, COUNT(*) as c
    FROM minute_data
    GROUP BY ticker, d
    HAVING c>=332
    ORDER BY d DESC
    LIMIT 20
    """
).fetchall()
print('sample completed (ticker,date,count):', rows[:5])
print('num completed rows =', len(rows))

for t in ['6954','6902','8058','2726','3382','3436']:
    r = cur.execute(
        "SELECT date(datetime), COUNT(*) FROM minute_data WHERE ticker=? GROUP BY date(datetime) ORDER BY 1 DESC LIMIT 3",
        (t,)
    ).fetchall()
    print(t, r)

conn.close()

