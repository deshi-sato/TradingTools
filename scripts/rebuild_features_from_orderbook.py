import sqlite3
from datetime import datetime

SRC = r"db\naut_market.db"
DST = r"db\naut_market_refeed.db"

def to_epoch(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("Z","")
        # 例: 2025-10-06T08:56:18.837
        return datetime.fromisoformat(s).timestamp()
    raise ValueError(f"unsupported ts: {v!r}")

conS = sqlite3.connect(SRC); conS.row_factory = sqlite3.Row
conD = sqlite3.connect(DST); conD.row_factory = sqlite3.Row
conD.execute("PRAGMA journal_mode=WAL;")
conD.execute("PRAGMA synchronous=NORMAL;")

conD.executescript("""
DROP TABLE IF EXISTS features_stream;
CREATE TABLE features_stream(
  symbol TEXT,
  t_exec REAL,
  ver INTEGER,
  f1 REAL, f2 REAL, f3 REAL, f4 REAL, f5 REAL, f6 REAL,
  score REAL,
  spread_ticks REAL,
  bid1 REAL, ask1 REAL,
  bidqty1 REAL, askqty1 REAL
);
""")

def tick(p): return 0.1

def compute_feats(r):
    bid1, ask1 = float(r["bid1"]), float(r["ask1"])
    buy_top3  = float(r["buy_top3"])  if "buy_top3"  in r.keys() else float(r["under_buy_qty"])
    sell_top3 = float(r["sell_top3"]) if "sell_top3" in r.keys() else float(r["over_sell_qty"])
    bidqty1, askqty1 = buy_top3, sell_top3

    den = (askqty1 + bidqty1) or 1.0
    f1 = (askqty1 - bidqty1) / den
    spr_ticks = (ask1 - bid1) / tick(ask1)
    f2 = spr_ticks
    f3 = bidqty1 / den
    f4 = f5 = f6 = 0.0

    score_raw = 0.5*f1 + 0.3*(-spr_ticks) + 0.2*(f3 - 0.5)*2
    score = max(0.0, min(10.0, 10.0*score_raw))
    return f1, f2, f3, f4, f5, f6, score, spr_ticks, bidqty1, askqty1, bid1, ask1

cur = conS.execute("""
SELECT ticker AS symbol, ts, bid1, ask1,
       over_sell_qty, under_buy_qty, sell_top3, buy_top3
FROM orderbook_snapshot ORDER BY ts ASC
""")

buf=[]; BATCH=1000
for r in cur:
    f1,f2,f3,f4,f5,f6,score,spr,bq,aq,b1,a1 = compute_feats(r)
    buf.append((r["symbol"], to_epoch(r["ts"]), 1,
                f1,f2,f3,f4,f5,f6, score, spr, b1,a1, bq,aq))
    if len(buf)>=BATCH:
        conD.executemany("INSERT INTO features_stream VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", buf)
        conD.commit(); buf.clear()
if buf:
    conD.executemany("INSERT INTO features_stream VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", buf)
    conD.commit()

print("rebuilt rows:", conD.execute("SELECT COUNT(*) FROM features_stream").fetchone()[0])
conS.close(); conD.close()
