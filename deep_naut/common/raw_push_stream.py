# -*- coding: utf-8 -*-
"""
common/raw_push_stream.py
PUSH(raw_push)行から特徴量を逐次生成して返す。
返却: (iterator[(feats:list[float], t_recv:float)], feature_names:list[str])

設計:
- 互換のため基礎20本は順序維持
- 追加7本は末尾に付与（ランナー側で先頭F本だけをモデル入力にスライス）
"""
import sqlite3, json, collections
from typing import Iterator, Tuple, Optional

SKIP_SPECIAL_SIGNS = True
SPECIAL_SIGNS = {"0102", "0108", "0118", "0119", "0120"}  # 特別/連続気配など

def _f(x):
    try: return float(x)
    except Exception: return 0.0

def iter_features_from_rawpush(db_path: str, table: str, symbol: str,
                               since_t: Optional[float]=None) -> Tuple[Iterator[tuple], list]:
    con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
    sql = f"SELECT t_recv, payload FROM {table} WHERE symbol=? ORDER BY t_recv ASC"
    params = [str(symbol)]
    if since_t is not None:
        sql = f"SELECT t_recv, payload FROM {table} WHERE symbol=? AND t_recv>? ORDER BY t_recv ASC"
        params.append(since_t)
    cur = con.execute(sql, params)

    base_names = [
        "price","mid","spread","imb","vwap_dev","vol_rate","val_rate",
        "depth_sell5","depth_buy5","microprice",
        "st_no","st_up","st_down",
        "d_bid_qty","d_ask_qty","is_trade","trade_size","imb_rate","spread_chg","mid_chg"
    ]
    extra_names = [
        "ret_5s","ret_10s","ret_accel_5s",
        "vol_ratio","vol_accel",
        "near_high_3m","status_onehot"  # onehot(UP/DOWN/NO)を1値にエンコード: +1/-1/0
    ]
    names = base_names + extra_names

    prev = {"tv":None,"val":None,"t":None,"bid_q":None,"ask_q":None,"imb":None,"spread":None,"mid":None}

    # 5秒/10秒/180秒用バッファ（tは秒単位）
    price_buf = collections.deque()     # (t_sec, price)
    volrate_buf = collections.deque()   # (t_sec, vol_rate)
    high_buf = collections.deque()      # (t_sec, price)

    def to_sec(ts):
        x = float(ts)
        if x > 1e12:   # ns
            return x/1e9
        elif x > 1e10: # ms
            return x/1000.0
        return x

    def prune(now_s):
        # 5s/10s参照用に 200秒だけ持てば十分
        cut = now_s - 190.0
        while price_buf and price_buf[0][0] < cut: price_buf.popleft()
        while volrate_buf and volrate_buf[0][0] < cut: volrate_buf.popleft()
        # ローリング高値は3分
        cut_h = now_s - 180.0
        while high_buf and high_buf[0][0] < cut_h: high_buf.popleft()

    def lookup_old(buf, target_s):
        # bufは昇順、target以前で一番近い値
        val = None
        for t_s, v in reversed(buf):
            if t_s <= target_s:
                val = v
                break
        return val

    def d1(curr, key):
        last = prev[key]
        diff = 0.0 if last is None else (curr - last)
        prev[key] = curr
        return diff

    def _iter():
        for r in cur:
            t_raw = float(r["t_recv"])
            t_s = to_sec(t_raw)
            d = json.loads(r["payload"])

            # 特別気配など除外
            if SKIP_SPECIAL_SIGNS:
                sb = str(d.get("BidSign") or "")
                sa = str(d.get("AskSign") or "")
                if sb in SPECIAL_SIGNS or sa in SPECIAL_SIGNS:
                    continue

            ask_p = _f(d.get("AskPrice"))   # 最良買気配
            bid_p = _f(d.get("BidPrice"))   # 最良売気配
            ask_q = _f(d.get("AskQty"))
            bid_q = _f(d.get("BidQty"))

            price = _f(d.get("CurrentPrice") or d.get("Price"))
            vwap  = _f(d.get("VWAP"))
            tv    = _f(d.get("TradingVolume"))
            val   = _f(d.get("TradingValue"))

            # 板深さ合計(1..5)
            depth_sell5 = depth_buy5 = 0.0
            for i in range(1,6):
                s = d.get(f"Sell{i}") or {}; b = d.get(f"Buy{i}") or {}
                depth_sell5 += _f(s.get("Qty")); depth_buy5 += _f(b.get("Qty"))

            # 派生
            mid    = (bid_p + ask_p)/2.0 if (bid_p>0 and ask_p>0) else 0.0
            spread = (ask_p - bid_p) if (ask_p>0 and bid_p>0) else 0.0
            den_q  = ask_q + bid_q
            imb    = (ask_q - bid_q)/den_q if den_q>0 else 0.0  # 買い-売り（修正済）
            vwap_dev   = (price - vwap)/vwap if vwap>0 else 0.0
            microprice = ((ask_p*bid_q + bid_p*ask_q)/den_q) if den_q>0 else 0.0

            # 差分速度
            dt = (t_s - prev["t"]) if prev["t"] is not None else 1.0
            prev["t"] = t_s
            d_bid_qty = d1(bid_q, "bid_q")
            d_ask_qty = d1(ask_q, "ask_q")
            d_tv      = d1(tv, "tv")
            d_val     = d1(val, "val")
            vol_rate  = max(0.0, d_tv) / max(1.0, dt)
            val_rate  = max(0.0, d_val) / max(1.0, dt)
            is_trade   = 1.0 if d_tv > 0 else 0.0
            trade_size = max(0.0, d_tv)
            imb_rate   = d1(imb, "imb")
            spread_chg = d1(spread, "spread")
            mid_chg    = d1(mid, "mid")

            # ステータス（最新版コード）
            st = str(d.get("CurrentPriceChangeStatus") or "")
            st_up   = 1.0 if st in ("0057","UP","up")     else 0.0
            st_down = 1.0 if st in ("0058","DOWN","down") else 0.0
            st_no   = 1.0 if (st_up==0.0 and st_down==0.0) else 0.0

            # ---- 追加7本 ----
            # バッファ更新＆掃除
            price_buf.append((t_s, price))
            volrate_buf.append((t_s, vol_rate))
            high_buf.append((t_s, price))
            prune(t_s)

            p_5 = lookup_old(price_buf, t_s - 5.0)
            p_10 = lookup_old(price_buf, t_s - 10.0)
            ret_5s  = (price / p_5 - 1.0) if p_5 and p_5>0 else 0.0
            ret_10s = (price / p_10 - 1.0) if p_10 and p_10>0 else 0.0

            # ret加速度（1秒前との差）
            p_1 = lookup_old(price_buf, t_s - 1.0)
            ret_1_old = (price / p_1 - 1.0) if p_1 and p_1>0 else 0.0
            p_6 = lookup_old(price_buf, t_s - 6.0)
            ret_5_old = (p_1 / p_6 - 1.0) if (p_1 and p_6 and p_6>0) else 0.0
            ret_accel_5s = ret_5s - ret_5_old

            # 出来高比（5秒MAで割る）
            vr_samples = [v for ts_, v in volrate_buf if ts_ >= t_s-5.0]
            ma5 = sum(vr_samples)/len(vr_samples) if vr_samples else 0.0
            vol_ratio = (vol_rate / ma5) if ma5>0 else 0.0
            # 出来高加速度（1秒差）
            vr_1 = lookup_old(volrate_buf, t_s - 1.0)
            vol_accel = (vol_rate - vr_1) if vr_1 is not None else 0.0

            # 直近3分の高値に対する位置
            hi_3m = max((px for _,px in high_buf), default=0.0)
            near_high_3m = (price/hi_3m - 1.0) if hi_3m>0 else 0.0

            # onehot簡略（+1:UP / -1:DOWN / 0:NO）
            status_onehot = (1.0 if st_up else (-1.0 if st_down else 0.0))

            base_feats = [
                price, mid, spread, imb, vwap_dev, vol_rate, val_rate,
                depth_sell5, depth_buy5, microprice,
                st_no, st_up, st_down,
                d_bid_qty, d_ask_qty, is_trade, trade_size, imb_rate, spread_chg, mid_chg
            ]
            extra_feats = [ret_5s, ret_10s, ret_accel_5s, vol_ratio, vol_accel, near_high_3m, status_onehot]
            feats = base_feats + extra_feats

            yield feats, t_raw
        con.close()

    return _iter(), names
