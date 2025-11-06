import sqlite3, json, numpy as np
from pathlib import Path

def _f(x):
    try: return float(x)
    except: return 0.0

def _load_raw(db, table, symbol):
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    cur = con.execute(f"SELECT t_recv, payload FROM {table} WHERE symbol=? ORDER BY t_recv ASC", (str(symbol),))
    rows=[]
    for r in cur:
        d = json.loads(r["payload"])
        rows.append({
            "t": float(r["t_recv"]),
            "ask_p": _f(d.get("AskPrice")), "bid_p": _f(d.get("BidPrice")),
            "ask_q": _f(d.get("AskQty")),   "bid_q": _f(d.get("BidQty")),
            "price": _f(d.get("CurrentPrice") or d.get("Price")),
            "vwap":  _f(d.get("VWAP")),
            "tv":    _f(d.get("TradingVolume")), "val": _f(d.get("TradingValue")),
            "st":    str(d.get("CurrentPriceChangeStatus") or ""),
            "bq": [ _f((d.get(f"Buy{i}")  or {}).get("Qty"))  for i in range(1,6) ],
            "sq": [ _f((d.get(f"Sell{i}") or {}).get("Qty")) for i in range(1,6) ],
        })
    con.close()
    return rows

def build_features(rows):
    # ベース指標
    t   = np.array([r["t"] for r in rows], dtype=np.float64)
    ask_p = np.array([r["ask_p"] for r in rows], dtype=np.float32)
    bid_p = np.array([r["bid_p"] for r in rows], dtype=np.float32)
    ask_q = np.array([r["ask_q"] for r in rows], dtype=np.float32)
    bid_q = np.array([r["bid_q"] for r in rows], dtype=np.float32)
    price = np.array([r["price"] for r in rows], dtype=np.float32)
    vwap  = np.array([r["vwap"]  for r in rows], dtype=np.float32)
    tv    = np.array([r["tv"]    for r in rows], dtype=np.float64)
    val   = np.array([r["val"]   for r in rows], dtype=np.float64)
    b5    = np.array([sum(r["bq"]) for r in rows], dtype=np.float32)
    s5    = np.array([sum(r["sq"]) for r in rows], dtype=np.float32)
    st    = np.array([r["st"] for r in rows])

    mid    = np.where((bid_p>0)&(ask_p>0), (bid_p+ask_p)/2, 0.0)
    spread = np.where((bid_p>0)&(ask_p>0), (ask_p-bid_p), 0.0)
    denom_q = bid_q + ask_q
    imb = np.divide((bid_q-ask_q), denom_q, out=np.zeros_like(bid_q), where=denom_q>0)
    vwap_dev = np.where(vwap>0, (price - vwap)/vwap, 0.0)
    microprice = np.divide((ask_p*bid_q + bid_p*ask_q), denom_q, out=np.zeros_like(bid_q), where=denom_q>0)

    # 変化量
    dt = np.maximum(1.0, np.diff(t, prepend=t[:1]))
    d_bid_qty = np.diff(bid_q, prepend=bid_q[:1])
    d_ask_qty = np.diff(ask_q, prepend=ask_q[:1])
    vol_rate  = np.maximum(0.0, np.diff(tv,  prepend=tv[:1]))/dt
    val_rate  = np.maximum(0.0, np.diff(val, prepend=val[:1]))/dt
    is_trade  = (np.diff(tv, prepend=tv[:1]) > 0).astype(np.float32)
    trade_sz  = np.maximum(0.0, np.diff(tv, prepend=tv[:1])).astype(np.float32)
    imb_rate  = np.diff(imb,    prepend=imb[:1])
    spread_chg= np.diff(spread, prepend=spread[:1])
    mid_chg   = np.diff(mid,    prepend=mid[:1])

    # ステータス one-hot
    up   = (st=="2") | (st=="Up")
    down = (st=="3") | (st=="Down")
    nochg= ~(up|down)
    st_onehot = np.stack([nochg.astype(np.float32), up.astype(np.float32), down.astype(np.float32)], axis=1)

    # 特徴を20本に
    X = np.column_stack([
        price, mid, spread, imb, vwap_dev, vol_rate, val_rate, s5, b5, microprice, st_onehot,
        d_bid_qty, d_ask_qty, is_trade, trade_sz, imb_rate, spread_chg, mid_chg
    ]).astype(np.float32)

    extras = {"mid": mid.astype(np.float32), "spread": spread.astype(np.float32), "t": t.astype(np.float64)}
    names = [
        "price","mid","spread","imb","vwap_dev","vol_rate","val_rate","depth_sell5","depth_buy5","microprice",
        "st_no","st_up","st_down","d_bid_qty","d_ask_qty","is_trade","trade_size","imb_rate","spread_chg","mid_chg"
    ]
    return X, extras, names

def main(db, table, symbol, out_npy="exports/features.npy", out_meta="exports/meta.npz", out_names="exports/feature_names.txt"):
    rows = _load_raw(db, table, symbol)
    if not rows: raise SystemExit("no rows")
    X, extras, names = build_features(rows)
    Path(out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, X)
    np.savez(out_meta, **extras)
    Path(out_names).write_text("\n".join(names), encoding="utf-8")
    print(f"OK: X{X.shape} -> {out_npy}; meta keys={list(extras.keys())}; names saved to {out_names}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="raw_push")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out-npy", default="exports/features.npy")
    ap.add_argument("--out-meta", default="exports/meta.npz")
    ap.add_argument("--out-names", default="exports/feature_names.txt")
    a=ap.parse_args()
    main(a.db, a.table, a.symbol, a.out_npy, a.out_meta, a.out_names)
