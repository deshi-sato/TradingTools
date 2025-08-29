# scripts/signal_scan.py
# -*- coding: utf-8 -*-
import argparse
import sqlite3
from datetime import datetime, timedelta, time
from pathlib import Path
import pandas as pd
import numpy as np
import re

JST_DATEFMT = "%Y-%m-%d"
JST_DATETIMEFMT = "%Y-%m-%d %H:%M:%S"

# ===== 可調整パラメータ =====
GAP_TH = 0.005  # 0.5% 以上を GU/GD とみなす
BRK_EPS = 0.0005  # 0.05% のバッファ
VOL_SPIKE_K = 3.0  # 1分出来高が直近20本MAの3倍以上
MA_WINDOW = 20
TRADING_SESSIONS = [
    (time(9, 0), time(11, 30)),
    (time(12, 30), time(15, 25)),
]

# ===== ニュース評価の辞書・重み =====
CATEGORY_WEIGHTS = {
    "材料": 2.0, "特報": 2.0, "決算": 1.6, "注目": 1.0,
    "市況": 0.5, "テク": 0.6, "速報": 1.0, "通貨": 0.5,
    "経済": 0.5, "業界": 0.6, "特集": 0.5, "総合": 0.4, "５％": 0.5,
}
POS_WORDS = ["上方修正","増配","最高益","上振れ","通期上方","自社株買い","大型受注","好調","過去最高"]
NEG_WORDS = ["下方修正","減配","赤字","下振れ","不適切会計","監理","公募増資","売出","未達","据え置き"]

CONS_POS = ["市場予想を上回る","コンセンサス超え","予想を上回る","上振れ","ガイダンス上方"]
CONS_NEG = ["市場予想を下回る","コンセンサス未達","予想を下回る","下振れ","据え置き","ガイダンス下方"]
STRONG_POS = ["大幅に上回る","大きく上回る","サプライズ上振れ"]
STRONG_NEG = ["大幅に下回る","大きく下回る","サプライズ下振れ"]

# ===== ユーティリティ =====
def ticker_candidates(ticker: str):
    """'3382.T' でも '3382' でもヒットするよう候補を返す"""
    m = re.match(r"^(\d{4})", str(ticker))
    code4 = m.group(1) if m else str(ticker)
    return code4 + ".T", code4  # (with_suffix, without_suffix)

def in_sessions(t: time) -> bool:
    for s, e in TRADING_SESSIONS:
        if t is not None and s <= t <= e:
            return True
    return False

def _norm_code4(ticker: str) -> str:
    """'7453.T' -> '7453' / 先頭4桁を抽出"""
    m = re.match(r"^(\d{4})", str(ticker))
    return m.group(1) if m else str(ticker)

def _to_yyyymmdd(date_str: str) -> str:
    s = str(date_str)
    if "-" in s:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")
    return s

def _pick_hhmm(x: str) -> str | None:
    """混在フォーマットから HH:MM を抜き出す（例: '25/08/25 05:35' -> '05:35'）"""
    if x is None: return None
    m = re.search(r"(\d{1,2}:\d{2})", str(x))
    return m.group(1) if m else None

def consensus_hint_from_text(text: str) -> int:
    """ニュース本文/タイトルからコンセンサス表現を±2〜±1で返す（1記事単位）"""
    if not isinstance(text, str) or not text:
        return 0
    score = 0
    if any(w in text for w in STRONG_POS): score += 2
    if any(w in text for w in STRONG_NEG): score -= 2
    if any(w in text for w in CONS_POS):   score += 1
    if any(w in text for w in CONS_NEG):   score -= 1
    return max(-2, min(2, score))

# ===== ニュース特徴量（kabutan_news.db 向け）=====
def load_news_features_kabutan(conn_news, ticker: str, date_str: str) -> dict:
    """
    当日(YYYYMMDD) + 前日(15:00-23:59) をスコア化して合算。
    news(url, date 'YYYYMMDD', time (混在可)), stock_mentions(news_url, code)
    """
    if conn_news is None:
        return {}
    code = _norm_code4(ticker)
    d8 = _to_yyyymmdd(date_str)
    prev_d8 = (datetime.strptime(d8, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")

    # 当日分 + 前日夕方(15:00-)分を一緒に取得
    q = """
    SELECT n.date, n.time, n.category, n.title, n.body
    FROM news n
    JOIN stock_mentions m ON m.news_url = n.url
    WHERE m.code = ?
      AND ( n.date = ?
            OR (n.date = ? AND n.time >= '15:00') )
    ORDER BY n.date ASC, n.time ASC
    """
    df = pd.read_sql(q, conn_news, params=[code, d8, prev_d8])

    if df.empty:
        return dict(
            news_cnt_total=0, news_cnt_preopen=0, news_cnt_am=0, news_cnt_pm=0,
            cat_weighted_score=0.0, lexi_sent_score=0.0, consensus_hint=0, news_score=0.0
        )

    # --- 時刻の混在フォーマット対応：time から HH:MM を抽出してからパース ---
    df = df.copy()
    df["_hhmm"] = df["time"].apply(_pick_hhmm)
    df.loc[df["_hhmm"].isna(), "_hhmm"] = "00:00"
    df["t"] = pd.to_datetime(df["_hhmm"], format="%H:%M", errors="coerce").dt.time
    # -------------------------------------------------------------------------

    # 前日夕方フラグ
    df["is_prev_evening"] = (df["date"] == prev_d8) & df["t"].apply(lambda tt: tt is not None and tt >= time(15,0))

    # 当日側の時間帯で集計
    is_today = (df["date"] == d8)
    def in_range(t, a, b): return (t is not None) and (a <= t <= b)
    today = df[is_today].copy()
    if not today.empty:
        preopen = today[today["t"].apply(lambda t: in_range(t, time(5,0),  time(8,59)))]
        am      = today[today["t"].apply(lambda t: in_range(t, time(9,0),  time(11,30)))]
        pm      = today[today["t"].apply(lambda t: in_range(t, time(12,30), time(15,25)))]
    else:
        preopen = am = pm = today

    # カテゴリ重み
    cat_score = float(sum(CATEGORY_WEIGHTS.get(str(c), 0.5) for c in df["category"].astype(str)))

    # 辞書スコア
    def lexi(text: str) -> int:
        if not isinstance(text, str): return 0
        s=0
        for w in POS_WORDS:
            if w in text: s+=1
        for w in NEG_WORDS:
            if w in text: s-=1
        return s
    lexi_sum = int(df.apply(lambda r: lexi(f"{r['title']} {r['body']}"), axis=1).sum())

    # コンセンサス表現（±2クリップ）
    hint_sum = int(df.apply(lambda r: consensus_hint_from_text(f"{r['title']} {r['body']}"), axis=1).sum())
    hint_sum = max(-2, min(2, hint_sum))

    # 寄り前ボーナス（当日の5:00–8:59）
    bonus = 0.2 * (0 if preopen is None else len(preopen))

    # 前日夕方のみ多い日はわずかに減衰
    ratio_prev = df["is_prev_evening"].mean() if len(df) else 0.0
    news_score = 0.6*cat_score + 0.4*lexi_sum + bonus + hint_sum
    news_score = news_score * (1.0 - 0.3*ratio_prev)  # 最大で0.3減衰

    return dict(
        news_cnt_total=int(len(df)),
        news_cnt_preopen=int(len(preopen) if len(today) else 0),
        news_cnt_am=int(len(am) if len(today) else 0),
        news_cnt_pm=int(len(pm) if len(today) else 0),
        cat_weighted_score=round(cat_score, 2),
        lexi_sent_score=float(lexi_sum),
        consensus_hint=int(hint_sum),
        news_score=round(float(news_score), 2),
    )

# ===== 株価側の読込 =====
def load_watchlist(path_watchlist: Path) -> pd.DataFrame:
    df = pd.read_csv(path_watchlist)
    assert "ticker" in df.columns, "watchlist に 'ticker' 列が必要です"
    if "side" not in df.columns:
        df["side"] = "BUY"
    return df[["ticker", "side"]].drop_duplicates()

def load_prev_ohlc(conn_daily, ticker, date_str):
    q = """
    SELECT date, open, high, low, close, volume
    FROM daily_data
    WHERE ticker = ? AND date < ?
    ORDER BY date DESC
    LIMIT 1
    """
    row = pd.read_sql(q, conn_daily, params=[ticker, date_str])
    return row.iloc[0] if not row.empty else None

def load_intraday(conn_min, ticker, date_str):
    t_with, t_plain = ticker_candidates(ticker)
    q = """
    SELECT datetime, open, high, low, close, volume
    FROM minute_data
    WHERE (ticker = ? OR ticker = ?) AND substr(datetime,1,10) = ?
    ORDER BY datetime ASC
    """
    df = pd.read_sql(q, conn_min, params=[t_with, t_plain, date_str])
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time"] = df["datetime"].dt.time
    df = df.dropna(subset=["open","high","low","close","volume"])
    df = df[df["time"].apply(in_sessions)]
    return df

# ===== シグナル判定 =====
def detect_signals_for_ticker(ticker, side, date_str, conn_daily, conn_min, conn_news=None):
    prev = load_prev_ohlc(conn_daily, ticker, date_str)
    intr = load_intraday(conn_min, ticker, date_str)
    if prev is None or intr.empty:
        return []

    # ニュース特徴量（銘柄×日で1回）
    news_feats = load_news_features_kabutan(conn_news, ticker, date_str) if conn_news else {}

    prev_close = float(prev["close"])
    prev_high  = float(prev["high"])
    prev_low   = float(prev["low"])

    # 当日寄りギャップ
    today_open = float(intr.iloc[0]["open"])
    gap_rate = today_open / prev_close - 1.0
    gu_gd = "GU" if gap_rate >= GAP_TH else ("GD" if gap_rate <= -GAP_TH else "NONE")

    # 出来高MA
    intr = intr.copy()
    intr["vol_ma"] = intr["volume"].rolling(MA_WINDOW, min_periods=3).mean()

    signals = []
    prev_high_up = prev_high * (1.0 + BRK_EPS)
    prev_low_dn  = prev_low  * (1.0 - BRK_EPS)

    for i in range(len(intr)):
        row = intr.iloc[i]
        tstr = row["datetime"].strftime(JST_DATETIMEFMT)
        last = float(row["close"])
        vol1 = float(row["volume"])
        vma  = float(row["vol_ma"]) if not np.isnan(row["vol_ma"]) else np.nan

        breakout = "UP" if last >= prev_high_up else ("DOWN" if last <= prev_low_dn else "NONE")
        vol_spike = bool(not np.isnan(vma) and vma > 0 and vol1 >= VOL_SPIKE_K * vma)

        if (gu_gd != "NONE" and i == 0) or breakout != "NONE" or vol_spike:
            reason = []
            if gu_gd != "NONE" and i == 0:
                reason.append(f"{gu_gd} {gap_rate:+.2%}")
            if breakout != "NONE":
                reason.append("前日高ブレイク" if breakout == "UP" else "前日安ブレイク")
            if vol_spike:
                reason.append(f"出来高急増 x{vol1/(vma+1e-9):.1f}")

            row_out = {
                "date": date_str, "ticker": ticker, "side": side, "t_detect": tstr,
                "gu_gd": gu_gd if i == 0 else "NONE", "breakout": breakout,
                "vol_spike": int(vol_spike), "last": last,
                "prev_close": prev_close, "prev_high": prev_high, "prev_low": prev_low,
                "vol_1m": int(vol1), "vol_ma20": (None if np.isnan(vma) else int(vma)),
                "reason": " / ".join(reason),
            }
            # ニュース特徴量を付与
            row_out.update({
                "news_score": news_feats.get("news_score", 0.0),
                "news_cnt_total": news_feats.get("news_cnt_total", 0),
                "news_cnt_preopen": news_feats.get("news_cnt_preopen", 0),
                "cat_weighted_score": news_feats.get("cat_weighted_score", 0.0),
                "lexi_sent_score": news_feats.get("lexi_sent_score", 0.0),
                "consensus_hint": news_feats.get("consensus_hint", 0),
            })
            signals.append(row_out)

    return signals

# ===== エントリーポイント =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w","--watchlist", required=True, help="watchlist_YYYY-MM-DD.csv")
    ap.add_argument("-d","--date", required=False, help="JST date YYYY-MM-DD（省略時は今日）")
    ap.add_argument("--dailydb", default="rss_daily.db")
    ap.add_argument("--minutedb", default="rss_data.db")
    ap.add_argument("--newsdb", default=None)
    ap.add_argument("-o","--out", required=False, help="signals_YYYY-MM-DD.csv")
    args = ap.parse_args()

    date_str = args.date or datetime.now().strftime(JST_DATEFMT)
    out_path = Path(args.out) if args.out else Path(f"signals_{date_str}.csv")

    wl = load_watchlist(Path(args.watchlist))
    conn_daily = sqlite3.connect(args.dailydb)
    conn_min   = sqlite3.connect(args.minutedb)
    conn_news  = sqlite3.connect(args.newsdb) if args.newsdb else None

    all_rows = []
    for _, r in wl.iterrows():
        all_rows += detect_signals_for_ticker(
            ticker=str(r["ticker"]), side=str(r["side"]),
            date_str=date_str, conn_daily=conn_daily, conn_min=conn_min, conn_news=conn_news
        )

    conn_daily.close()
    conn_min.close()
    if conn_news: conn_news.close()

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "date","ticker","side","t_detect","gu_gd","breakout","vol_spike",
            "last","prev_close","prev_high","prev_low","vol_1m","vol_ma20","reason",
            "news_score","news_cnt_total","news_cnt_preopen","cat_weighted_score","lexi_sent_score","consensus_hint"
        ])
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Wrote: {out_path.resolve()} (rows={len(df)})")

if __name__ == "__main__":
    main()
