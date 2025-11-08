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
from typing import Iterator, Tuple, Optional, Dict

class Rolling3:
    __slots__ = ("buf", "total")
    def __init__(self):
        self.buf = collections.deque(maxlen=3)
        self.total = 0.0
    def push(self, x: float | None) -> float:
        if x is None:
            x = 0.0
        if len(self.buf) == 3:
            self.total -= self.buf[0]
        self.buf.append(x)
        self.total += x
        return self.mean
    @property
    def mean(self) -> float:
        n = len(self.buf)
        return self.total / n if n else 0.0

rolling_price: Dict[str, Rolling3] = {}
rolling_vol: Dict[str, Rolling3] = {}
rolling_imb: Dict[str, Rolling3] = {}
symbol_state: Dict[str, Dict[str, float]] = collections.defaultdict(dict)

def get_roll(store: Dict[str, Rolling3], symbol: str) -> Rolling3:
    if symbol not in store:
        store[symbol] = Rolling3()
    return store[symbol]

def safe_div(a: float, b: float, eps: float = 1.0) -> float:
    return a / (b if b > 0 else eps)

def clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

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
    rolling_names = ["price_ma3","vol_ma3","imb_ma3","candle_up"]
    names = base_names + extra_names + rolling_names

    prev = {"tv":None,"val":None,"t":None,"bid_q":None,"ask_q":None,"imb":None,"spread":None,"mid":None}

    # 5秒/10秒/180秒用バッファ（tは秒単位）
    price_buf = collections.deque()     # (t_sec, price)
    volspeed_buf = collections.deque()  # (t_sec, vol_speed)
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
        while volspeed_buf and volspeed_buf[0][0] < cut: volspeed_buf.popleft()
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
            symbol_id = str(d.get("Symbol") or symbol)
            state = symbol_state.setdefault(symbol_id, {})

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
            vol_speed = max(0.0, d_tv) / max(1.0, dt)
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

            price_now = price
            vol_now = trade_size
            imb_now = imb
            price_prev = float(state.get("price_prev", price_now))
            vol_prev = float(state.get("vol_prev", max(vol_now - 1.0, 0.0)))
            vol_rate = clip(safe_div(vol_now, vol_prev), 0.0, 10.0)
            roll_price = get_roll(rolling_price, symbol_id)
            roll_vol = get_roll(rolling_vol, symbol_id)
            roll_imb = get_roll(rolling_imb, symbol_id)
            price_ma3 = roll_price.push(price_now)
            vol_ma3 = roll_vol.push(vol_now)
            imb_ma3 = roll_imb.push(imb_now)
            is_up = (price_now > price_prev) or (st in ("+", "005", "010", "015", "0057", "UP", "up"))
            candle_up = 1.0 if is_up else 0.0
            state["price_prev"] = price_now
            state["vol_prev"] = vol_now

            # ---- 追加7本 ----
            # バッファ更新＆掃除
            price_buf.append((t_s, price))
            volspeed_buf.append((t_s, vol_speed))
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
            vr_samples = [v for ts_, v in volspeed_buf if ts_ >= t_s-5.0]
            ma5 = sum(vr_samples)/len(vr_samples) if vr_samples else 0.0
            vol_ratio = (vol_speed / ma5) if ma5>0 else 0.0
            # 出来高加速度（1秒差）
            vr_1 = lookup_old(volspeed_buf, t_s - 1.0)
            vol_accel = (vol_speed - vr_1) if vr_1 is not None else 0.0

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
            new_feats = [price_ma3, vol_ma3, imb_ma3, candle_up]
            feats = base_feats + extra_feats + new_feats

            yield feats, t_raw
        con.close()

    return _iter(), names

def ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    """
    ����SQLite���̃e�[�u���ɏo�͑��̂��ꂩ�����邩���`�F�b�N���ăm�点�������f�����B
    columns: {"col_name": "REAL", ...}
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    have = {row[1] for row in cur.fetchall()}
    for name, decl in columns.items():
        if name not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()

def export_features_to_table(
    db_path: str,
    table_src: str,
    table_dst: str,
    symbol: str,
    batch_size: int = 512,
) -> None:
    """
    PUSH(raw_push)�ϐ��̃e�[�u����iter_features_from_rawpush�Ŏ擾���ăt�B�[�`���X�e�b�v���V�[�����ɏ����߂�.
    - table_dst �͏����� DROP -> CREATE ���s����
    - batch_size ����INSERT�����g���ŗ���ݒ�
    """
    rows, names = iter_features_from_rawpush(db_path, table_src, symbol)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_dst}")
        col_defs = ", ".join(f"\"{n}\" REAL" for n in names)
        cur.execute(
            f"""
            CREATE TABLE {table_dst} (
                symbol TEXT NOT NULL,
                t_exec INTEGER NOT NULL,
                {col_defs}
            )
            """
        )
        conn.commit()

        insert_cols = ", ".join(["symbol", "t_exec"] + [f"\"{n}\"" for n in names])
        placeholders = ", ".join(["?"] * (len(names) + 2))
        insert_sql = f"INSERT INTO {table_dst} ({insert_cols}) VALUES ({placeholders})"

        buf: list[list] = []
        total = 0
        for feats, t_raw in rows:
            record = [str(symbol), int(float(t_raw))] + [float(x) for x in feats]
            buf.append(record)
            if len(buf) >= batch_size:
                cur.executemany(insert_sql, buf)
                conn.commit()
                total += len(buf)
                buf.clear()
        if buf:
            cur.executemany(insert_sql, buf)
            conn.commit()
            total += len(buf)
        print(f"[export_features_to_table] {table_dst} <- {table_src} ({symbol}): {total} rows")
    finally:
        conn.close()
