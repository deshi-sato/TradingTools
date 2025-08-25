"""
pick_tickers.py
  戦略タグ (T{window}_V{volma}) を実際の銘柄へ落とすスクリプト。
  - DB: rss_daily.db / table: daily_bars(ticker,date,open,high,low,close,volume, PRIMARY KEY(ticker,date))
  - スコア: 直近 T 本の log(close) に対する線形回帰の傾き（モメンタム）
  - 出来高フィルタ: 直近 V 本の出来高移動平均 (vol_maV) > --min-vol-ma
  - 出力: date, side, tag, ticker, score, ret_T, vol_maV, close, さらにサイズ列
      * size_mode: 'atr' or 'sigma'（既定 atr）
      * size_atr / size_sigma を常に出力（数値）。'size' 列は size_mode に応じた採用値。

サイズ計算:
  ATR方式:
    - ATR = rolling_mean( TrueRange, atr_window )
    - TrueRange = max( H-L, |H-PrevClose|, |L-PrevClose| )
    - 想定損切り幅 = atr_mult * ATR
    - 1トレードの許容損失 = capital * risk_pct
    - 株数 = floor( (capital * risk_pct) / (atr_mult * ATR) )
  σ方式:
    - σ = 直近 T 本の logリターン標準偏差
    - 目標日次変動額 = capital * vol_target_pct
    - 株数 = floor( (capital * vol_target_pct) / (σ * close) )

CLI:
  --db               rss_daily.db のパス（既定: rss_daily.db）
  --tag              例: T180_V30   ※ --from-daily-pick が無い場合は必須
  --asof             YYYY-MM-DD。未指定なら DB の max(date)
  --top-long         BUY 上位の件数（既定: 1）
  --top-short        SELL 上位の件数（既定: 1）
  --from-daily-pick  data/analysis/daily_strategy_pick.csv の最新行 buy/sell から選定
  --out              出力CSV。未指定なら data/analysis/picks_<asof>.csv
  --min-vol-ma       出来高V日平均の下限。既定 0.0（=無効）
  --min-days         1銘柄あたり必要本数の下限。既定 None（= max(T,V)+1）
  --size-mode        'atr' or 'sigma'（既定 'atr'）
  --capital          運用資金（既定 1000000.0）
  --risk-pct         ATR方式: 1トレード許容損失率（既定 0.005 = 0.5%）
  --atr-window       ATRの窓（既定 14）
  --atr-mult         損切り幅のATR係数（既定 1.5）
  --vol-target-pct   σ方式: 想定日次変動の目標率（既定 0.01 = 1%）
"""

from __future__ import annotations
import argparse
import re
import sqlite3
from math import floor, isfinite
from pathlib import Path

import numpy as np
import pandas as pd


TAG_RE = re.compile(r"^T(?P<T>\d+)_V(?P<V>\d+)$")


def parse_tag(tag: str) -> tuple[int, int]:
    m = TAG_RE.match(str(tag))
    if not m:
        raise ValueError(f"Invalid tag format: {tag!r} (expected like T180_V30)")
    T = int(m.group("T"))
    V = int(m.group("V"))
    return T, V


def latest_asof(conn: sqlite3.Connection) -> str:
    row = pd.read_sql("SELECT max(date) AS d FROM daily_bars", conn)
    if row.empty or pd.isna(row["d"].iloc[0]):
        raise RuntimeError("daily_bars has no rows (cannot determine asof).")
    return str(row["d"].iloc[0])


def trend_slope(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    if len(y) < 5 or np.any(~np.isfinite(y)):
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def compute_atr(df: pd.DataFrame, window: int) -> float:
    """df: 必須列 ['high','low','close']、日付昇順で直近まで含む想定。"""
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(
        axis=1
    )
    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else float("nan")


def compute_sigma(close: pd.Series, T: int) -> float:
    """直近T本の logリターン標準偏差（≒日次σ）。"""
    px = pd.to_numeric(close, errors="coerce")
    logret = np.log(px / px.shift(1)).iloc[-T:]
    sigma = logret.std(ddof=1)
    return float(sigma) if pd.notna(sigma) else float("nan")


def safe_floor(x: float) -> int:
    try:
        if not isfinite(x):
            return 0
        return max(0, floor(x))
    except Exception:
        return 0


def round_lot(size: int, lot: int = 100) -> int:
    """売買単位で丸め（例: 100株単位）。0未満は0。"""
    if size <= 0:
        return 0
    return (int(size) // int(lot)) * int(lot)


def select_for_tag(
    conn: sqlite3.Connection,
    tag: str,
    asof: str,
    top_long: int,
    top_short: int,
    min_vol_ma: float,
    min_days: int | None,
    size_mode: str,
    capital: float,
    risk_pct: float,
    atr_window: int,
    atr_mult: float,
    vol_target_pct: float,
    lot: int,
    min_notional: float,
    max_notional: float,
) -> pd.DataFrame:
    T, V = parse_tag(tag)

    # 必要列を全部取る（サイズ計算のため H/L/O も使う）
    df = pd.read_sql(
        "SELECT ticker, date, open, high, low, close, volume "
        "FROM daily_bars WHERE date <= ? ORDER BY ticker, date",
        conn,
        params=[asof],
    )
    if df.empty:
        raise RuntimeError(f"No rows in DB for asof<={asof}")

    rows: list[dict] = []
    n_total = n_short = n_vol = n_nan = 0

    for tkr, g in df.groupby("ticker", sort=False):
        n_total += 1
        g = g.sort_values("date")
        need = max(T, V) + 1 if min_days is None else int(min_days)
        if len(g) < need:
            n_short += 1
            continue

        vol = pd.to_numeric(g["volume"], errors="coerce")
        vol_ma = vol.rolling(V).mean().iloc[-1]
        if not np.isfinite(vol_ma) or float(vol_ma) <= float(min_vol_ma):
            n_vol += 1
            continue

        close = pd.to_numeric(g["close"], errors="coerce")
        cT = close.iloc[-T:]
        slope = trend_slope(cT)
        if not np.isfinite(slope):
            n_nan += 1
            continue

        # スコア補助
        ret_T = float(cT.iloc[-1] / cT.iloc[0] - 1.0)
        last_close = float(close.iloc[-1])

        # === サイズ計算 ===
        atr = compute_atr(
            g.iloc[-(max(T, atr_window) + 1) :][["high", "low", "close"]].assign(
                high=pd.to_numeric(g["high"], errors="coerce").iloc[
                    -(max(T, atr_window) + 1) :
                ],
                low=pd.to_numeric(g["low"], errors="coerce").iloc[
                    -(max(T, atr_window) + 1) :
                ],
                close=pd.to_numeric(g["close"], errors="coerce").iloc[
                    -(max(T, atr_window) + 1) :
                ],
            ),
            atr_window,
        )
        sigma = compute_sigma(close, T)

        # ATR方式
        size_atr = 0
        if np.isfinite(atr) and atr > 0:
            stop_width = atr_mult * atr
            denom = stop_width if stop_width > 0 else np.nan
            size_atr = safe_floor((capital * risk_pct) / denom)

        # σ方式
        size_sigma = 0
        if np.isfinite(sigma) and sigma > 0 and last_close > 0:
            size_sigma = safe_floor((capital * vol_target_pct) / (sigma * last_close))

        # 採用サイズ（生値 → 単元丸め → 代金フィルタ）
        size_raw = int(size_atr if size_mode == "atr" else size_sigma)
        size = round_lot(size_raw, lot)
        notional = float(last_close * size)

        # 代金レンジ（0 は無効）
        if size <= 0:
            eligible = False
        else:
            eligible = True
            if min_notional > 0 and notional < min_notional:
                eligible = False
            if max_notional > 0 and notional > max_notional:
                eligible = False

        if not eligible:
            # フィルタで除外：候補に入れない
            continue

        rows.append(
            {
                "ticker": tkr,
                "score": float(slope),
                "ret_T": ret_T,
                "vol_maV": float(vol_ma),
                "close": last_close,
                "atr": float(atr) if np.isfinite(atr) else np.nan,
                "sigma": float(sigma) if np.isfinite(sigma) else np.nan,
                "size_raw": int(size_raw),
                "size_atr": int(size_atr),  # ← 追加
                "size_sigma": int(size_sigma),  # ← 追加
                "size": int(size),
                "notional": float(notional),
                "size_mode": size_mode,
            }
        )

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError(
            "No candidates after filters (empty set) | "
            f"tickers={n_total}, too_short={n_short}, vol_ma<={min_vol_ma}={n_vol}, slope_nan={n_nan}"
        )

    long_df = res.sort_values("score", ascending=False).head(max(top_long, 0)).copy()
    long_df["side"] = "BUY"
    short_df = res.sort_values("score", ascending=True).head(max(top_short, 0)).copy()
    short_df["side"] = "SELL"

    both = pd.concat([long_df, short_df], ignore_index=True)
    both["tag"] = tag
    both["date"] = asof

    cols = [
        "date",
        "side",
        "tag",
        "ticker",
        "score",
        "ret_T",
        "vol_maV",
        "close",
        "atr",
        "sigma",
        "size_raw",
        "size_atr",
        "size_sigma",
        "size",
        "notional",
        "size_mode",
    ]
    return both[cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="rss_daily.db")
    ap.add_argument("--tag")
    ap.add_argument("--asof")
    ap.add_argument("--top-long", type=int, default=1)
    ap.add_argument("--top-short", type=int, default=1)
    ap.add_argument("--from-daily-pick")
    ap.add_argument("--out")
    ap.add_argument("--min-vol-ma", type=float, default=0.0)
    ap.add_argument("--min-days", type=int, default=None)
    # サイズ計算の引数
    ap.add_argument("--size-mode", choices=["atr", "sigma"], default="atr")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--risk-pct", type=float, default=0.005)
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--atr-mult", type=float, default=1.5)
    ap.add_argument("--vol-target-pct", type=float, default=0.01)
    ap.add_argument("--lot", type=int, default=100, help="売買単位（既定100株）")
    ap.add_argument(
        "--min-notional", type=float, default=0.0, help="最小売買代金（0で無効）"
    )
    ap.add_argument(
        "--max-notional", type=float, default=0.0, help="最大売買代金（0で無効）"
    )
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    try:
        asof = args.asof or latest_asof(con)

        def run(one_tag: str) -> pd.DataFrame:
            return select_for_tag(
                con,
                one_tag,
                asof,
                top_long=args.top_long,
                top_short=args.top_short,
                min_vol_ma=args.min_vol_ma,
                min_days=args.min_days,
                size_mode=args.size_mode,
                capital=args.capital,
                risk_pct=args.risk_pct,
                atr_window=args.atr_window,
                atr_mult=args.atr_mult,
                vol_target_pct=args.vol_target_pct,
                lot=args.lot,
                min_notional=args.min_notional,
                max_notional=args.max_notional,
            )

        # ここからが正しい分岐
        if args.from_daily_pick:
            dp = pd.read_csv(args.from_daily_pick)
            if dp.empty:
                raise RuntimeError("daily_strategy_pick.csv is empty")
            row = dp.iloc[-1]
            if not args.asof:
                asof = str(row["date"])
            buy_df = run(str(row["buy"])).query("side=='BUY'")
            sell_df = run(str(row["sell"])).query("side=='SELL'")
            out_df = pd.concat([buy_df, sell_df], ignore_index=True)
        else:
            if not args.tag:
                raise ValueError("Either --tag or --from-daily-pick is required.")
            out_df = run(args.tag)

        out_path = args.out
        if not out_path:
            Path("data/analysis").mkdir(parents=True, exist_ok=True)
            out_path = f"data/analysis/picks_{asof}.csv"

        out_df.to_csv(out_path, index=False)
        print(f"[INFO] Wrote {out_path} ({len(out_df)} rows)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
