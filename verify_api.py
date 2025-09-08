\"\"\"verify_api.py : Auto-generated placeholder

- file: verify_api.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
import sys, types

# Lightweight stubs for heavy optional deps used by score_table
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

from app import app

def main():
    with app.test_client() as c:
        res = c.get('/api/snapshots?limit=5')
        print('status:', res.status_code)
        if not res.is_json:
            print('Not JSON response')
            print(res.data[:200])
            return
        data = res.get_json()
        print('rows:', len(data))
        # Print first 3 rows keys and ticker/score
        for i, row in enumerate(data[:3]):
            print(i, 'keys=', list(row.keys()))
            print('  ticker=', row.get('ticker'), 'score=', row.get('score'))

if __name__ == '__main__':
    main()
