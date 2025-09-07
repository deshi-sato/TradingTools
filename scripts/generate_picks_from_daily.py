import argparse
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# ====== ユーティリティ ======
def read_daily(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT ticker, date, open, high, low, close, volume FROM daily_bars",
        con, parse_dates=["date"]
    )
    con.close()
    return df.sort_values(["ticker","date"]).reset_index(drop=True)

def rsi(series: pd.Series, n=3) -> pd.Series:
    diff = series.diff()
    up = diff.clip(lower=0).rolling(n).mean()
    dn = (-diff.clip(upper=0)).rolling(n).mean()
    rs = up / (dn.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def add_features(g: pd.DataFrame) -> pd.DataFrame:
    # MAs
    g["ma5"]  = g["close"].rolling(5).mean()
    g["ma25"] = g["close"].rolling(25).mean()
    g["ma75"] = g["close"].rolling(75).mean()
    # Volume MA
    g["vma20"]= g["volume"].rolling(20).mean()
    # 前日値
    g["prev_close"] = g["close"].shift(1)
    g["prev_high"]  = g["high"].shift(1)
    g["prev_low"]   = g["low"].shift(1)
    # 出来高比・代金
    g["vol_ratio"] = g["volume"] / g["vma20"]
    g["turnover"]  = g["close"] * g["volume"]
    # ローソク形状
    rng = (g["high"] - g["low"]).replace(0, np.nan)
    g["pos_in_range"] = (g["close"] - g["low"]) / rng  # 0=安値側,1=高値側
    body = (g["close"] - g["open"]).abs()
    upper_shadow = g["high"] - g[["close","open"]].max(axis=1)
    lower_shadow = g[["close","open"]].min(axis=1) - g["low"]
    eps = 1e-9
    g["upper_wick_ratio"] = upper_shadow / (body + eps)
    g["lower_wick_ratio"] = lower_shadow / (body + eps)
    # RSI(3)
    g["rsi3"] = rsi(g["close"], 3)
    return g

# ====== スコア（“前日”に評価する内容）======
def score_buy_row(r: pd.Series) -> float:
    s = 0.0
    # 上昇トレンド（強）＋レンジ上部
    s += 2.0 * ((r["ma5"] > r["ma25"]) and (r["ma25"] > r["ma75"]))
    s += 1.0 * (pd.notna(r["pos_in_range"]) and (r["pos_in_range"] >= 0.8))
    # 出来高
    if pd.notna(r["vol_ratio"]):
        if r["vol_ratio"] >= 1.2: s += 1.0
        if r["vol_ratio"] >= 2.0: s += 1.0
    return float(s)

def score_sell_row(r: pd.Series) -> float:
    s = 0.0
    # 下降トレンド（強）＋レンジ下部
    s += 2.0 * ((r["ma5"] < r["ma25"]) and (r["ma25"] < r["ma75"]))
    s += 1.5 * (pd.notna(r["pos_in_range"]) and (r["pos_in_range"] <= 0.2))
    # 出来高
    if pd.notna(r["vol_ratio"]):
        if r["vol_ratio"] >= 1.5: s += 1.0
        if r["vol_ratio"] >= 2.0: s += 1.0
    return float(s)

# ====== 指数トレンド判定（任意）======
def trend_up_idx(idx_row: pd.DataFrame) -> bool:
    return bool((idx_row["ma5"].iloc[-1] > idx_row["ma25"].iloc[-1]) and (idx_row["ma25"].iloc[-1] > idx_row["ma75"].iloc[-1]))

def trend_down_idx(idx_row: pd.DataFrame) -> bool:
    c = idx_row["close"].iloc[-1]; ma5 = idx_row["ma5"].iloc[-1]; ma25 = idx_row["ma25"].iloc[-1]
    return bool((c < ma5) and (ma5 < ma25))

# ====== picks 生成（未来参照なし）======
def generate_picks(
    df: pd.DataFrame,
    start: date,
    end: date,
    out_dir: Path,
    min_turnover: float,
    topn: int,
    index_ticker: str|None,
    disable_sell_in_uptrend: bool,
    buy_overbought: float,
    sell_oversold: float,
    upper_wick_ratio_thr: float,
    lower_wick_ratio_thr: float,
    # Optional: weights-based scoring and single CSV output
    w_trend=None,
    w_volume=None,
    w_momo=None,
    single_out_csv=None,
):
    # ウォームアップを確保
    start_buf = start - timedelta(days=150)
    df = df[(df["date"] >= pd.Timestamp(start_buf)) & (df["date"] <= pd.Timestamp(end))].copy()

    # 特徴量
    df = pd.concat((add_features(g) for _, g in df.groupby("ticker", sort=False)), ignore_index=True)

    # 当日スコア → 前日にシフト（＝翌営業日のシグナルに使う）
    buy_today  = pd.concat((g.apply(score_buy_row, axis=1)  for _, g in df.groupby("ticker", sort=False)))
    sell_today = pd.concat((g.apply(score_sell_row, axis=1) for _, g in df.groupby("ticker", sort=False)))
    buy_today.index = df.index; sell_today.index = df.index

    df["score_buy_prev"]  = df.groupby("ticker")[buy_today.name if buy_today.name else "close"].transform(lambda x: 0)  # ダミー列確保
    df["score_sell_prev"] = df.groupby("ticker")[sell_today.name if sell_today.name else "close"].transform(lambda x: 0)
    df["score_buy_prev"]  = buy_today.groupby(df["ticker"]).shift(1)
    df["score_sell_prev"] = sell_today.groupby(df["ticker"]).shift(1)

    # 前日ベースのリバウンド除外フラグ
    df["turnover_prev"] = df.groupby("ticker")["turnover"].shift(1)
    df["rsi3_prev"]     = df.groupby("ticker")["rsi3"].shift(1)
    df["upper_wick_prev"]= df.groupby("ticker")["upper_wick_ratio"].shift(1)
    df["lower_wick_prev"]= df.groupby("ticker")["lower_wick_ratio"].shift(1)
    df["pos_prev"]       = df.groupby("ticker")["pos_in_range"].shift(1)

    # 営業日リスト & 翌日マップ
    days = sorted(df["date"].dropna().unique())
    next_map = {days[i]: (days[i+1] if i+1 < len(days) else None) for i in range(len(days))}

    # 指数（任意）
    idx_df = None
    if index_ticker:
        idx_df = df[df["ticker"] == index_ticker].loc[:, ["date","close","ma5","ma25","ma75"]].copy()

    # Aggregate rows across days if single_out_csv is specified
    all_rows = []

    for d, daydf in df.groupby("date"):
        next_d = next_map.get(d)
        if next_d is None:
            continue
        # 出力する日付が期間内か
        if not (pd.Timestamp(start) <= pd.Timestamp(next_d) <= pd.Timestamp(end)):
            continue

        # 地合いフィルタ
        allow_sell = True
        if idx_df is not None:
            idx_day = idx_df[idx_df["date"] == d]
            if not idx_day.empty and disable_sell_in_uptrend and trend_up_idx(idx_day):
                allow_sell = False

        # 前日データが揃っていて、かつ前日流動性クリアな銘柄
        cands = daydf.dropna(subset=[
            "score_buy_prev","score_sell_prev","turnover_prev",
            "rsi3_prev","upper_wick_prev","lower_wick_prev","pos_prev"
        ]).copy()
        cands = cands[cands["turnover_prev"] >= float(min_turnover)]

        # ==== リバウンドリスク除外 ====
        # BUY除外：前日が過熱（RSI高）、長い上ヒゲ、レンジ上過ぎ（過ぎたるは及ばざる…）
        buy_mask = (
            (cands["rsi3_prev"] <= buy_overbought) &
            (cands["upper_wick_prev"] <= upper_wick_ratio_thr) &
            (cands["pos_prev"] <= 0.98)  # 完全張り付きに近い極端は避ける
        )
        # SELL除外：前日が売られすぎ（RSI低）、長い下ヒゲ、レンジ下過ぎ
        sell_mask = (
            (cands["rsi3_prev"] >= sell_oversold) &
            (cands["lower_wick_prev"] <= lower_wick_ratio_thr) &
            (cands["pos_prev"] >= 0.02)
        )

        rows = []

        # BUY上位
        # Optional weighted ranking using linear combination of components
        use_weighted = ((w_trend is not None) and (w_volume is not None) and (w_momo is not None))
        if use_weighted:
            # Component scores (prev day)
            cands["trend_up_prev"] = ((cands["ma5"] > cands["ma25"]) & (cands["ma25"] > cands["ma75"])).astype(float)
            cands["vol_comp_prev"] = (cands["vol_ratio"] / 2.0).clip(lower=0, upper=1)
            cands["momo_buy_prev"] = ((cands["rsi3_prev"] - 50.0) / 50.0).clip(lower=0, upper=1)
            cands["score_buy_weighted"] = (
                (w_trend * cands["trend_up_prev"]) + (w_volume * cands["vol_comp_prev"]) + (w_momo * cands["momo_buy_prev"])
            )
            buy_cands = cands[buy_mask].sort_values(["score_buy_weighted","turnover_prev"], ascending=[False, False]).head(topn)
        else:
            buy_cands = cands[buy_mask].sort_values(["score_buy_prev","turnover_prev"], ascending=[False, False]).head(topn)
        for _, r in buy_cands.iterrows():
            if r["score_buy_prev"] > 0:
                score_val = (float(r["score_buy_weighted"]) if use_weighted else float(r["score_buy_prev"]))
                rows.append({"date": pd.Timestamp(next_d).strftime('%Y-%m-%d'), "side":"BUY","code":r["ticker"],"score":round(score_val,6)})

        # SELL上位（許可された日だけ）
        if allow_sell:
            if use_weighted:
                cands["trend_dn_prev"] = ((cands["ma5"] < cands["ma25"]) & (cands["ma25"] < cands["ma75"])).astype(float)
                cands["momo_sell_prev"] = ((50.0 - cands["rsi3_prev"]) / 50.0).clip(lower=0, upper=1)
                cands["score_sell_weighted"] = (
                    (w_trend * cands["trend_dn_prev"]) + (w_volume * cands["vol_comp_prev"]) + (w_momo * cands["momo_sell_prev"])
                )
                sell_cands = cands[sell_mask].sort_values(["score_sell_weighted","turnover_prev"], ascending=[False, False]).head(topn)
            else:
                sell_cands = cands[sell_mask].sort_values(["score_sell_prev","turnover_prev"], ascending=[False, False]).head(topn)
            for _, r in sell_cands.iterrows():
                if r["score_sell_prev"] > 0:
                    score_val = (float(r["score_sell_weighted"]) if use_weighted else float(r["score_sell_prev"]))
                    rows.append({"date": pd.Timestamp(next_d).strftime('%Y-%m-%d'), "side":"SELL","code":r["ticker"],"score":round(score_val,6)})

        if rows:
            if single_out_csv is not None:
                all_rows.extend(rows)
            else:
                out_path = out_dir / f"picks_{pd.Timestamp(next_d).strftime('%Y-%m-%d')}.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows, columns=["side","code","score"]).to_csv(out_path, index=False, encoding="utf-8-sig")

    # Write aggregated single CSV if requested
    if single_out_csv is not None:
        Path(single_out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows, columns=["date","side","code","score"]).to_csv(single_out_csv, index=False, encoding="utf-8-sig")

# ====== エントリーポイント ======
def main():
    ap = argparse.ArgumentParser(description="日足のみで picks を“未来参照なし”生成（リバウンド除外入り）")
    ap.add_argument("--db-path", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out-dir", default="./data/analysis")
    ap.add_argument("--min-turnover", type=float, default=200_000_000)
    ap.add_argument("--topn", type=int, default=1)
    ap.add_argument("--index-ticker", default="1306.T", help="指数ティッカー（例: 1306.T TOPIXETF / 1321.T 日経ETF）空文字で無効")
    ap.add_argument("--disable-sell-in-uptrend", action="store_true", help="指数が上昇トレンドの日はSELLを出さない")
    # リバウンド除外の閾値（調整可能）
    ap.add_argument("--buy-overbought", type=float, default=90.0, help="BUY除外: 前日RSI(3)がこの値を超えたら除外")
    ap.add_argument("--sell-oversold",  type=float, default=10.0, help="SELL除外: 前日RSI(3)がこの値未満なら除外")
    ap.add_argument("--upper-wick-ratio", type=float, default=1.5, help="BUY除外: 前日の上ヒゲ/実体 比がこの値超で除外")
    ap.add_argument("--lower-wick-ratio", type=float, default=1.5, help="SELL除外: 前日の下ヒゲ/実体 比がこの値超で除外")
    # Additional CLI for weighted scoring & single CSV output
    ap.add_argument("--w-trend", type=float, default=None)
    ap.add_argument("--w-volume", type=float, default=None)
    ap.add_argument("--w-momo", type=float, default=None)
    ap.add_argument("--out", type=str, default=None, help="単一のpicks CSV出力先。指定時は期間を通じた集約CSVを出力")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    df = read_daily(args.db_path)
    generate_picks(
        df=df, start=start, end=end, out_dir=Path(args.out_dir),
        min_turnover=args.min_turnover, topn=args.topn,
        index_ticker=(None if str(args.index_ticker).strip().lower() in ["", "none"] else args.index_ticker),
        disable_sell_in_uptrend=args.disable_sell_in_uptrend,
        buy_overbought=args.buy_overbought,
        sell_oversold=args.sell_oversold,
        upper_wick_ratio_thr=args.upper_wick_ratio,
        lower_wick_ratio_thr=args.lower_wick_ratio,
        w_trend=args.w_trend,
        w_volume=args.w_volume,
        w_momo=args.w_momo,
        single_out_csv=(Path(args.out) if args.out else None),
    )

if __name__ == "__main__":
    main()
