\"\"\"inspect_db.py : Auto-generated placeholder

- file: inspect_db.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
from pathlib import Path
import sqlite3
import sys, types

# Stub heavy optional deps to allow importing app/score_table
mods = {
    'xlwings': types.ModuleType('xlwings'),
    'pandas': types.ModuleType('pandas'),
    'mplfinance': types.ModuleType('mplfinance'),
    'matplotlib': types.ModuleType('matplotlib'),
    'matplotlib.pyplot': types.ModuleType('matplotlib.pyplot'),
    'matplotlib.dates': types.ModuleType('matplotlib.dates'),
}
mods['matplotlib.pyplot'].rcParams = {}
mods['mplfinance'].make_addplot = lambda *a, **k: None
for name, mod in mods.items():
    sys.modules.setdefault(name, mod)

from app import DB_PATH

def main():
    db = Path(DB_PATH)
    print('DB_PATH =', db)
    if not db.exists():
        print('DB not found')
        return
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print('tables:', tables)
    def safe_count(table):
        try:
            c = cur.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        except Exception as e:
            print(f'count({table}) error:', e)
            return None
        print(f'rows({table}) =', c)
        return c
    safe_count('quote_latest')
    safe_count('minute_data')

    # Pick a few tickers and inspect minute bars
    tickers = [r[0] for r in cur.execute('SELECT ticker FROM quote_latest LIMIT 5').fetchall()]
    print('sample tickers:', tickers)
    for t in tickers:
        row = cur.execute("SELECT date(datetime) FROM minute_data WHERE ticker=? ORDER BY datetime DESC LIMIT 1", (t,)).fetchone()
        print('ticker', t, 'latest date =', row[0] if row else None)
        if row and row[0]:
            cnt = cur.execute("SELECT COUNT(*) FROM minute_data WHERE ticker=? AND date(datetime)=?", (t, row[0])).fetchone()[0]
            print('  bars today =', cnt)
    conn.close()

if __name__ == '__main__':
    main()
