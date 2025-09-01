# -*- coding: utf-8 -*-
"""
cls
  rss_index.db（指標）と rss_snapshot.db（当日1分足）を読み、
  買い/売りのエントリー候補（IFDOCO用）をCSVに出力します。

想定スキーマ：
- rss_snapshot.db
  * today_data(ticker, datetime, open, high, low, close, volume) ... 当日のみ格納されている想定
  * （任意）watchlist / snapshot_watchlist 等に side(BUY/SELL) があれば優先的に絞り込み

- rss_index.db
  * テーブル名は固定でなくてもOK。ticker と score_buy / score_sell があるテーブルを自動探索します。

※ 可能な限りロバストに設計。存在しない列・テーブルは自動スキップ。
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import tz


JST = tz.gettz("Asia/Tokyo")


# ============ ユーザー調整しやすい閾値 ============
VOL_SPIKE_MULT = 1.5  # 出来高SMA20に対する倍率
VWAP_MIN_BARS_BELOW = 3  # VWAP下（上）にいた最小本数（リクレイム/フェイル判定用）
R1_MULT = 1.3  # 1st 目標 = 1.3R
R2_MULT = 2.0  # 2nd 目標 = 2.0R
MIN_R_YEN = 5.0  # 損切り幅の下限（小さすぎるノイズ回避）
ORB_START = time(9, 0, 0)  # ORB開始
ORB_END = time(9, 15, 0)  # ORB終了（この時間は含めない）
# ==============================================


@dataclass
class Signal:
    time: pd.Timestamp
    ticker: str
    side: str  # BUY / SELL
    strategy: str  # BUY_ORB / BUY_VWAP_RECLAIM / SELL_ORB / SELL_VWAP_FAIL
    entry: float
    stop: float
    target1: float
    target2: float
    R: float
    price: float
    vwap: float
    ma5: float
    ma25: float
    vol: float
    vol_sma20: float
    atr1: float
    pre_score_buy: Optional[float]
    pre_score_sell: Optional[float]
    notes: str


def consec_true(series: pd.Series) -> pd.Series:
    """Trueが何本連続しているか（当該バーでの長さ）"""
    out = np.zeros(len(series), dtype=int)
    v = series.to_numpy(dtype=bool)
    cnt = 0
    for i in range(len(v)):
        cnt = cnt + 1 if v[i] else 0
        out[i] = cnt
    return pd.Series(out, index=series.index, dtype=int)


def load_today_data(
    conn_snap: sqlite3.Connection,
    tickers: Optional[List[str]] = None,
    start_hhmm: str = "09:00",
) -> pd.DataFrame:
    if "today_data" not in list_tables(conn_snap):
        raise RuntimeError("rss_snapshot.db に today_data テーブルが見つかりません。")

    base = pd.read_sql("SELECT * FROM today_data", conn_snap)
    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
    base = base.dropna(subset=["datetime"]).copy()
    if len(base) == 0:
        return base

    # 最新営業日だけ
    latest_date = base["datetime"].dt.date.max()
    base = base[base["datetime"].dt.date == latest_date].copy()

    # ★ 当日 9:00 以降だけに限定（start_hhmmで可変）
    hh, mm = map(int, start_hhmm.split(":"))
    date_part = pd.Timestamp(latest_date)
    market_open = pd.Timestamp.combine(date_part, time(hh, mm))
    base = base[base["datetime"] >= market_open].copy()

    if tickers:
        base = base[base["ticker"].isin(tickers)].copy()

    for c in ["open", "high", "low", "close", "volume"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.dropna(subset=["open", "high", "low", "close", "volume"])
    base = base.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    return base


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-db", required=False, help="rss_index.db のパス（省略可）")
    ap.add_argument("--snapshot-db", required=True, help="rss_snapshot.db のパス")
    ap.add_argument("--out", default="./out/signals_latest.csv", help="出力CSVパス")
    ap.add_argument(
        "--log", default="./logs/signal_watcher.log", help="ログファイルパス"
    )
    ap.add_argument(
        "--start-hhmm", default="09:00", help="監視開始時刻（当日基準, 例: 09:00）"
    )
    ap.add_argument("--deadline-min", type=int, default=60, help="デッドライン（分）")
    ap.add_argument(
        "--snapshot-only",
        action="store_true",
        help="index/newsが無い場合はsnapshotのみで判定",
    )
    return ap.parse_args()


def ensure_dirs(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def log(msg: str, log_path: str) -> None:
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    ensure_dirs(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)


def list_tables(conn: sqlite3.Connection) -> List[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table'"
    return [r[0] for r in conn.execute(q).fetchall()]


def table_has_columns(conn: sqlite3.Connection, table: str, cols: List[str]) -> bool:
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {r[1] for r in info}  # cid, name, type, notnull, dflt, pk
    return all(c in names for c in cols)


def find_score_table(conn_index: sqlite3.Connection) -> Optional[str]:
    """
    rss_index.db 内で 'ticker' と 'score_buy' or 'score_sell' を持つテーブルを探索。
    最初に見つかったテーブルを返す。
    """
    for t in list_tables(conn_index):
        if not table_has_columns(conn_index, t, ["ticker"]):
            continue
        info = conn_index.execute(f"PRAGMA table_info({t})").fetchall()
        names = {r[1] for r in info}
        if "score_buy" in names or "score_sell" in names:
            return t
    return None


def maybe_load_scores(conn_index: Optional[sqlite3.Connection]) -> pd.DataFrame:
    if conn_index is None:
        return pd.DataFrame(columns=["ticker", "score_buy", "score_sell"])
    t = find_score_table(conn_index)
    if t is None:
        return pd.DataFrame(columns=["ticker", "score_buy", "score_sell"])
    df = pd.read_sql(f"SELECT * FROM {t}", conn_index)
    keep = [c for c in df.columns if c in ("ticker", "score_buy", "score_sell")]
    return df[keep].drop_duplicates("ticker")


def find_watchlist(conn_snap: sqlite3.Connection) -> Optional[str]:
    """
    snapshot側に watchlist的テーブルがあるなら使う（side列でBUY/SELLを事前指定できる）。
    候補名をいくつか試す。
    """
    candidates = [
        "watchlist",
        "snapshot_watchlist",
        "watchlist_today",
        "watchlist_preopen",
    ]
    for t in list_tables(conn_snap):
        if t.lower() in candidates and table_has_columns(conn_snap, t, ["ticker"]):
            return t
    return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: 単一tickerの当日1分足（ソート済）
    返り値: 各種指標列を追加したDataFrame
    """
    out = df.copy()
    # MA
    out["ma5"] = out["close"].rolling(5).mean()
    out["ma25"] = out["close"].rolling(25).mean()

    # VWAP (典型価格でなくcloseベースでも実務上十分。ここはtypicalにしておく)
    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    cum_v = out["volume"].cumsum()
    out["vwap"] = (tp * out["volume"]).cumsum() / cum_v.replace(0, np.nan)

    # 出来高SMA20
    out["vol_sma20"] = out["volume"].rolling(20).mean()

    # ATR(14) on 1-min
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr1"] = tr.rolling(14).mean()

    # ORB用レンジ（9:00–9:15）
    hh, ll = calc_orb_levels(out)
    out["orb_high"] = hh
    out["orb_low"] = ll

    return out


def calc_orb_levels(df: pd.DataFrame) -> Tuple[float, float]:
    d = df.copy()
    t = d["datetime"].dt.time
    mask = (t >= ORB_START) & (t < ORB_END)
    if mask.any():
        rng = d.loc[mask, ["high", "low"]]
        return float(rng["high"].max()), float(rng["low"].min())
    # 初動データがない場合はNAN返し
    return np.nan, np.nan


def last_cross_up(a: pd.Series, b: pd.Series) -> bool:
    """直近2本で a が b を上抜いたか？"""
    if len(a) < 2 or len(b) < 2:
        return False
    return bool((a.iloc[-2] <= b.iloc[-2]) and (a.iloc[-1] > b.iloc[-1]))


def last_cross_down(a: pd.Series, b: pd.Series) -> bool:
    """直近2本で a が b を下抜いたか？"""
    if len(a) < 2 or len(b) < 2:
        return False
    return bool((a.iloc[-2] >= b.iloc[-2]) and (a.iloc[-1] < b.iloc[-1]))


def bars_below(a: pd.Series, b: pd.Series, n: int) -> int:
    """直近連続して a<b だった本数を返す"""
    cnt = 0
    for x, y in zip(a[::-1], b[::-1]):
        if pd.isna(x) or pd.isna(y):
            break
        if x < y:
            cnt += 1
        else:
            break
    return cnt


def bars_above(a: pd.Series, b: pd.Series, n: int) -> int:
    cnt = 0
    for x, y in zip(a[::-1], b[::-1]):
        if pd.isna(x) or pd.isna(y):
            break
        if x > y:
            cnt += 1
        else:
            break
    return cnt


def swing_low_recent(df: pd.DataFrame, lookback: int = 3) -> float:
    lo = df["low"].tail(lookback + 1).min()
    return float(lo)


def swing_high_recent(df: pd.DataFrame, lookback: int = 3) -> float:
    hi = df["high"].tail(lookback + 1).max()
    return float(hi)


def propose_oco(
    side: str, entry_price: float, ref_df: pd.DataFrame
) -> Tuple[float, float, float, float, str]:
    """
    損切り/目標の自動提案
    - side: BUY/SELL
    - entry_price: シグナル発生時の価格
    - ref_df: 直近データ（指標列含む）
    """
    atr = float(ref_df["atr1"].iloc[-1]) if not ref_df["atr1"].isna().all() else np.nan
    orb_h = float(ref_df["orb_high"].iloc[-1])
    orb_l = float(ref_df["orb_low"].iloc[-1])

    note_parts = []

    if side == "BUY":
        sl1 = swing_low_recent(ref_df, lookback=3)
        sl2 = orb_l if not np.isnan(orb_l) else sl1
        atr_stop = entry_price - max(atr * 0.8 if not np.isnan(atr) else 0.0, MIN_R_YEN)
        stop = min(sl2, atr_stop)  # より“遠い”方にしてもOKだが、まずは保守的に
        if stop >= entry_price:
            stop = entry_price - max(
                MIN_R_YEN, (atr * 0.8 if not np.isnan(atr) else MIN_R_YEN)
            )
        note_parts.append("stop=max(swingLow3, ORB_L, ATR*0.8) 採用")
        R = entry_price - stop
        t1 = entry_price + R * R1_MULT
        t2 = entry_price + R * R2_MULT

    else:  # SELL
        sh1 = swing_high_recent(ref_df, lookback=3)
        sh2 = orb_h if not np.isnan(orb_h) else sh1
        atr_stop = entry_price + max(atr * 0.8 if not np.isnan(atr) else 0.0, MIN_R_YEN)
        stop = max(sh2, atr_stop)
        if stop <= entry_price:
            stop = entry_price + max(
                MIN_R_YEN, (atr * 0.8 if not np.isnan(atr) else MIN_R_YEN)
            )
        note_parts.append("stop=max(swingHigh3, ORB_H, ATR*0.8) 採用")
        R = stop - entry_price
        t1 = entry_price - R * R1_MULT
        t2 = entry_price - R * R2_MULT

    return float(stop), float(t1), float(t2), float(R), "; ".join(note_parts)


def detect_signals_one(
    df: pd.DataFrame, pre_buy: Optional[float], pre_sell: Optional[float]
) -> List[Signal]:
    """
    単一ticker（当日分）でシグナル検出（バーごとに評価）。
    条件を満たした各バーの時刻で Signal を作成する。
    """
    sigs: List[Signal] = []
    if len(df) < 30:
        return sigs

    d = compute_indicators(df)

    close = d["close"]
    vwap = d["vwap"]
    ma5 = d["ma5"]
    ma25 = d["ma25"]
    vol = d["volume"]
    vol_s = d["vol_sma20"]
    orb_h = d["orb_high"]
    orb_l = d["orb_low"]

    # ===== ORB =====
    # BUY: ORB高値上抜け + VWAP上 + ma5>ma25 + 出来高スパイク
    mask_buy_orb = (
        (~orb_h.isna())
        & (close > orb_h)
        & (close > vwap)
        & (ma5 > ma25)
        & (~vol_s.isna())
        & (vol > vol_s * VOL_SPIKE_MULT)
    )

    # SELL: ORB安値下抜け + VWAP下 + ma5<ma25 + 出来高スパイク
    mask_sell_orb = (
        (~orb_l.isna())
        & (close < orb_l)
        & (close < vwap)
        & (ma5 < ma25)
        & (~vol_s.isna())
        & (vol > vol_s * VOL_SPIKE_MULT)
    )

    # ===== VWAP cross（連続カウントで代替）=====
    below = close < vwap
    above = close > vwap
    below_run = consec_true(below)
    above_run = consec_true(above)

    cross_up = (close.shift(1) <= vwap.shift(1)) & (close > vwap)  # 直近で下→上
    cross_down = (close.shift(1) >= vwap.shift(1)) & (close < vwap)  # 直近で上→下

    mask_buy_reclaim = (
        cross_up
        & (below_run.shift(1, fill_value=0) >= VWAP_MIN_BARS_BELOW)
        & (ma5 > ma25)
    )
    mask_sell_fail = (
        cross_down
        & (above_run.shift(1, fill_value=0) >= VWAP_MIN_BARS_BELOW)
        & (ma5 < ma25)
    )

    # ====== それぞれの真のバーをSignal化 ======
    def emit(idx: int, side: str, strat: str):
        # その時点までのデータでOCOを提案
        ref = d.iloc[: idx + 1]
        price = float(close.iloc[idx])
        stop, t1, t2, R, note = propose_oco(side, price, ref)
        sigs.append(
            Signal(
                time=d["datetime"].iloc[idx],
                ticker=str(d["ticker"].iloc[idx]),
                side=side,
                strategy=strat,
                entry=price,
                stop=stop,
                target1=t1,
                target2=t2,
                R=R,
                price=price,
                vwap=float(vwap.iloc[idx]),
                ma5=float(ma5.iloc[idx]) if not pd.isna(ma5.iloc[idx]) else np.nan,
                ma25=float(ma25.iloc[idx]) if not pd.isna(ma25.iloc[idx]) else np.nan,
                vol=int(vol.iloc[idx]),
                vol_sma20=(
                    float(vol_s.iloc[idx]) if not pd.isna(vol_s.iloc[idx]) else np.nan
                ),
                atr1=(
                    float(ref["atr1"].iloc[-1])
                    if not ref["atr1"].isna().all()
                    else np.nan
                ),
                pre_score_buy=pre_buy,
                pre_score_sell=pre_sell,
                notes=note,
            )
        )

    for i in np.where(mask_buy_orb.to_numpy())[0]:
        emit(int(i), "BUY", "BUY_ORB")
    for i in np.where(mask_buy_reclaim.to_numpy())[0]:
        emit(int(i), "BUY", "BUY_VWAP_RECLAIM")
    for i in np.where(mask_sell_orb.to_numpy())[0]:
        emit(int(i), "SELL", "SELL_ORB")
    for i in np.where(mask_sell_fail.to_numpy())[0]:
        emit(int(i), "SELL", "SELL_VWAP_FAIL")

    return sigs


def main():
    args = parse_args()
    ensure_dirs(args.out)
    ensure_dirs(args.log)

    log("=== signal_watcher start ===", args.log)
    log(f"snapshot_db={args.snapshot_db}", args.log)
    if args.index_db:
        log(f"index_db={args.index_db}", args.log)
    log(f"start_hhmm={args.start_hhmm}, deadline_min={args.deadline_min}", args.log)

    # DB接続
    conn_idx = (
        sqlite3.connect(args.index_db)
        if (args.index_db and not args.snapshot_only and os.path.exists(args.index_db))
        else None
    )
    conn_snp = sqlite3.connect(args.snapshot_db)

    try:
        df_scores = maybe_load_scores(conn_idx)
        log(f"score table rows = {len(df_scores)}", args.log)

        wl_table = find_watchlist(conn_snp)
        wl = None
        if wl_table:
            try:
                wl = pd.read_sql(f"SELECT * FROM {wl_table}", conn_snp)
                log(f"watchlist loaded: table={wl_table}, rows={len(wl)}", args.log)
            except Exception as e:
                log(f"watchlist load error: {e}", args.log)

        tickers: Optional[List[str]] = None
        side_pref: Dict[str, str] = {}
        if wl is not None and "ticker" in wl.columns:
            tickers = sorted(wl["ticker"].astype(str).unique().tolist())
            if "side" in wl.columns:
                tmp = wl[["ticker", "side"]].dropna()
                side_pref = {
                    str(r.ticker): str(r.side).upper()
                    for r in tmp.itertuples(index=False)
                }
                log(f"side preferences found for {len(side_pref)} tickers", args.log)

        # ★ 9:00以降へ限定読み込み
        base = load_today_data(conn_snp, tickers, start_hhmm=args.start_hhmm)
        # ★ 9:00～10:00 未満だけを扱う（10:00 以降は読み込まない＝カット）
        hh, mm = map(int, args.start_hhmm.split(":"))
        latest_date = base["datetime"].dt.date.max()
        window_start = pd.Timestamp.combine(pd.Timestamp(latest_date), time(hh, mm))
        window_end = window_start + pd.Timedelta(minutes=args.deadline_min)  # 例: 10:00

        # ここで最終トリミング：10:00 以降は除外
        before = len(base)
        base = base[
            (base["datetime"] >= window_start) & (base["datetime"] < window_end)
        ].copy()
        after = len(base)
        log(
            f"window={window_start.time()}-{window_end.time()} rows: {before}->{after} (cut >= end)",
            args.log,
        )
        if len(base) == 0:
            log("today_data（9:00以降）が空です。", args.log)
            pd.DataFrame(
                columns=[
                    "time",
                    "ticker",
                    "side",
                    "strategy",
                    "entry",
                    "stop",
                    "target1",
                    "target2",
                    "R",
                    "price",
                    "vwap",
                    "ma5",
                    "ma25",
                    "vol",
                    "vol_sma20",
                    "atr1",
                    "pre_score_buy",
                    "pre_score_sell",
                    "notes",
                ]
            ).to_csv(args.out, index=False, encoding="utf-8")
            return

        # tickersを確定（watchlistが無ければtoday_dataから）
        if tickers is None:
            tickers = sorted(base["ticker"].astype(str).unique().tolist())
            log(f"tickers from today_data: {len(tickers)}銘柄", args.log)

        # scores を dict化
        score_buy_map = {
            str(r.ticker): float(r.score_buy)
            for r in df_scores.itertuples(index=False)
            if "score_buy" in df_scores.columns and pd.notna(r.score_buy)
        }
        score_sell_map = {
            str(r.ticker): float(r.score_sell)
            for r in df_scores.itertuples(index=False)
            if "score_sell" in df_scores.columns and pd.notna(r.score_sell)
        }

        all_sigs: List[Signal] = []

        for tk in tickers:
            df_t = base[base["ticker"] == tk].copy()
            if len(df_t) < 30:
                continue

            pre_b = score_buy_map.get(tk)
            pre_s = score_sell_map.get(tk)

            sigs = detect_signals_one(df_t, pre_b, pre_s)

            # watchlistで side 指定があればフィルタリング
            if tk in side_pref:
                pref = side_pref[tk]
                sigs = [s for s in sigs if s.side.upper() == pref.upper()]

            all_sigs.extend(sigs)

        # 出力
        if len(all_sigs) == 0:
            log("シグナルなし。空のCSVを出力します。", args.log)
            pd.DataFrame(
                columns=[
                    "time",
                    "ticker",
                    "side",
                    "strategy",
                    "entry",
                    "stop",
                    "target1",
                    "target2",
                    "R",
                    "price",
                    "vwap",
                    "ma5",
                    "ma25",
                    "vol",
                    "vol_sma20",
                    "atr1",
                    "pre_score_buy",
                    "pre_score_sell",
                    "notes",
                ]
            ).to_csv(args.out, index=False, encoding="utf-8")
            return

        # --- 出力 ---
        rows = []
        for s in all_sigs:
            rows.append(
                {
                    "time": s.time,
                    "ticker": s.ticker,
                    "side": s.side,
                    "strategy": s.strategy,
                    "entry": round(s.entry, 3),
                    "stop": round(s.stop, 3),
                    "target1": round(s.target1, 3),
                    "target2": round(s.target2, 3),
                    "R": round(s.R, 3),
                    "price": round(s.price, 3),
                    "vwap": round(s.vwap, 3),
                    "ma5": round(s.ma5, 3) if pd.notna(s.ma5) else np.nan,
                    "ma25": round(s.ma25, 3) if pd.notna(s.ma25) else np.nan,
                    "vol": int(s.vol),
                    "vol_sma20": (
                        round(s.vol_sma20, 3) if pd.notna(s.vol_sma20) else np.nan
                    ),
                    "atr1": round(s.atr1, 3) if pd.notna(s.atr1) else np.nan,
                    "pre_score_buy": (
                        s.pre_score_buy if s.pre_score_buy is not None else np.nan
                    ),
                    "pre_score_sell": (
                        s.pre_score_sell if s.pre_score_sell is not None else np.nan
                    ),
                    "notes": s.notes,
                }
            )
        df_out = pd.DataFrame(rows)

        # 優先度スコア（簡易）：pre_score + 0.3*出来高倍率 - 0.1*|R-10| + 0.2*(ORBならボーナス)
        def _prio(r):
            pre = r["pre_score_buy"] if r["side"] == "BUY" else r["pre_score_sell"]
            pre = 0.0 if pd.isna(pre) else float(pre)
            ratio = (
                (float(r["vol"]) / float(r["vol_sma20"]))
                if pd.notna(r["vol_sma20"]) and r["vol_sma20"] > 0
                else 1.0
            )
            rterm = -0.1 * abs(float(r["R"]) - 10.0)  # R=10円を理想に
            sbonus = 0.2 if "ORB" in str(r["strategy"]) else 0.0
            return pre + 0.3 * ratio + rterm + sbonus

        df_out["priority"] = df_out.apply(_prio, axis=1)
        df_out["rank_in_side"] = (
            df_out.groupby("side")["priority"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )

        # 表示しやすい並び（まず時刻、その後ticker/side）
        df_out = df_out.sort_values(["time", "ticker", "side"]).reset_index(drop=True)

        # ★ Excel向け：BOM付きUTF-8で保存（日本語noteの文字化け回避）
        df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
        log(f"出力: {args.out}（{len(df_out)}件）", args.log)

    finally:
        if conn_idx is not None:
            conn_idx.close()
        conn_snp.close()
        log("=== signal_watcher end ===", args.log)


if __name__ == "__main__":
    main()
