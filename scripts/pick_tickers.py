# -*- coding: utf-8 -*-
"""
戦略タグ (T{window}_V{volma}) を実銘柄へ落とす。
- DB: rss_daily.db / table: daily_bars(ticker,date,open,high,low,close,volume, PRIMARY KEY(ticker,date))
- スコア: 直近 T 本の log(close) に対する線形回帰の傾き（モメンタム）
- 出来高フィルタ: 直近 V 本の出来高移動平均 (vol_maV) > --min-vol-ma
- 出力: date, side, tag, ticker, score, ret_T, vol_maV, close, さらにサイズ列
    * size_mode: 'atr' or 'sigma'（既定 atr）
    * size_atr / size_sigma は常に出力。'size' は size_mode に応じた採用値。

今回の拡張:
- --log          実行ログ（.log）を出力（コンソールにも出す）
- --log-level    INFO/DEBUG など
- --explain-out  採否理由付きの全銘柄明細CSVを出力（eligible/reason を含む）
- ログ内容: 実行パラメータ、母集団/通過数/落選数、主な落選理由TOP、BUY/SELL採用件数と重複数、上位明細
"""

from __future__ import annotations
import argparse
import logging
import re
import sqlite3
from math import floor, isfinite
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


TAG_RE = re.compile(r"^T(?P<T>\d+)_V(?P<V>\d+)$")


# -------------------- ロギング --------------------
def setup_logger(log_path: str | None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("picks")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# -------------------- ユーティリティ --------------------
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
    y = np.log(pd.to_numeric(prices, errors="coerce").values)
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
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
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


# -------------------- 本体ロジック --------------------
def build_base_table(
    conn: sqlite3.Connection,
    tag: str,
    asof: str,
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
    """
    全銘柄について指標・サイズを計算し、「eligible/reason」を付けた明細テーブル(DataFrame)を返す。
    ここでは落選も含めて全て残す（ログ/--explain-out 用）。
    """
    T, V = parse_tag(tag)

    df = pd.read_sql(
        "SELECT ticker, date, open, high, low, close, volume "
        "FROM daily_bars WHERE date <= ? ORDER BY ticker, date",
        conn,
        params=[asof],
    )
    if df.empty:
        raise RuntimeError(f"No rows in DB for asof<={asof}")

    rows: list[dict] = []

    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        need = max(T, V) + 1 if min_days is None else int(min_days)

        reason: list[str] = []
        eligible = True

        if len(g) < need:
            eligible = False
            reason.append("len<need")

        vol = pd.to_numeric(g["volume"], errors="coerce")
        vol_ma = vol.rolling(V).mean().iloc[-1]
        if not np.isfinite(vol_ma) or float(vol_ma) <= float(min_vol_ma):
            eligible = False
            reason.append(f"vol_ma<={min_vol_ma}")

        close = pd.to_numeric(g["close"], errors="coerce")
        cT = close.iloc[-T:]
        slope = trend_slope(cT)
        if not np.isfinite(slope):
            eligible = False
            reason.append("slope=NaN")

        # 補助値
        ret_T = float(cT.iloc[-1] / cT.iloc[0] - 1.0) if len(cT) >= 2 and np.isfinite(cT.iloc[0]) else np.nan
        last_close = float(close.iloc[-1]) if len(close) else np.nan

        # === サイズ計算 ===
        sub = g.iloc[-(max(T, atr_window) + 1) :].copy()
        atr = compute_atr(sub[["high", "low", "close"]], atr_window)
        sigma = compute_sigma(close, T)

        size_atr = 0
        if np.isfinite(atr) and atr > 0:
            stop_width = atr_mult * atr
            denom = stop_width if stop_width > 0 else np.nan
            size_atr = safe_floor((capital * risk_pct) / denom)

        size_sigma = 0
        if np.isfinite(sigma) and sigma > 0 and last_close > 0:
            size_sigma = safe_floor((capital * vol_target_pct) / (sigma * last_close))

        size_raw = int(size_atr if size_mode == "atr" else size_sigma)
        size = round_lot(size_raw, lot)
        notional = float(last_close * size) if size > 0 and np.isfinite(last_close) else 0.0

        if size <= 0:
            eligible = False
            reason.append("size<=0")
        if min_notional > 0 and notional < min_notional:
            eligible = False
            reason.append(f"notional<{min_notional}")
        if max_notional > 0 and notional > max_notional:
            eligible = False
            reason.append(f"notional>{max_notional}")

        rows.append(
            {
                "ticker": tkr,
                "score": float(slope) if np.isfinite(slope) else np.nan,
                "ret_T": ret_T,
                "vol_maV": float(vol_ma) if np.isfinite(vol_ma) else np.nan,
                "close": float(last_close) if np.isfinite(last_close) else np.nan,
                "atr": float(atr) if np.isfinite(atr) else np.nan,
                "sigma": float(sigma) if np.isfinite(sigma) else np.nan,
                "size_raw": int(size_raw),
                "size_atr": int(size_atr),
                "size_sigma": int(size_sigma),
                "size": int(size),
                "notional": float(notional),
                "size_mode": size_mode,
                "eligible": bool(eligible),
                "reason": ";".join(reason) if reason else "",
            }
        )

    base = pd.DataFrame(rows).sort_values("ticker")
    return base


def choose_buy_sell(
    base: pd.DataFrame, tag: str, asof: str, top_long: int, top_short: int
) -> pd.DataFrame:
    """
    eligible==True のみから BUY/SELL を抽出し、最終の出力フォーマットで返す。
    """
    eligible = base.query("eligible==True").copy()
    if eligible.empty:
        raise RuntimeError("No candidates after filters (eligible==False for all).")

    buy = eligible.sort_values("score", ascending=False).head(max(top_long, 0)).copy()
    buy["side"] = "BUY"

    sell = eligible.sort_values("score", ascending=True).head(max(top_short, 0)).copy()
    sell["side"] = "SELL"

    both = pd.concat([buy, sell], ignore_index=True)
    both["tag"] = tag
    both["date"] = asof

    cols = [
        "date", "side", "tag", "ticker",
        "score", "ret_T", "vol_maV", "close",
        "atr", "sigma", "size_raw", "size_atr", "size_sigma",
        "size", "notional", "size_mode",
    ]
    return both[cols]


# -------------------- CLI --------------------
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
    ap.add_argument("--min-notional", type=float, default=0.0, help="最小売買代金（0で無効）")
    ap.add_argument("--max-notional", type=float, default=0.0, help="最大売買代金（0で無効）")
    # 拡張: ログ/明細
    ap.add_argument("--log", help="実行ログの出力先 .log（未指定ならコンソールのみ）")
    ap.add_argument("--log-level", default="INFO", help="INFO/DEBUG etc.")
    ap.add_argument("--explain-out", help="採否理由付きの明細CSV（eligible/reason含む）")
    args = ap.parse_args()

    logger = setup_logger(args.log, args.log_level)

    con = sqlite3.connect(args.db)
    try:
        asof = args.asof or latest_asof(con)

        def run(one_tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
            base = build_base_table(
                con, one_tag, asof,
                args.min_vol_ma, args.min_days,
                args.size_mode, args.capital, args.risk_pct,
                args.atr_window, args.atr_mult, args.vol_target_pct,
                args.lot, args.min_notional, args.max_notional
            )
            picks = choose_buy_sell(base, one_tag, asof, args.top_long, args.top_short)
            return picks, base

        # buy/sell タグの決定
        if args.from_daily_pick:
            dp = pd.read_csv(args.from_daily_pick)
            if dp.empty:
                raise RuntimeError("daily_strategy_pick.csv is empty")
            row = dp.iloc[-1]
            if not args.asof:
                asof = str(row["date"])

            buy_df, base_b = run(str(row["buy"]))
            sell_df, base_s = run(str(row["sell"]))

            out_df = pd.concat([buy_df.query("side=='BUY'"),
                                sell_df.query("side=='SELL'")], ignore_index=True)
            base = pd.concat([base_b.assign(tag=str(row["buy"])),
                              base_s.assign(tag=str(row["sell"]))],
                             ignore_index=True)
            tag_info = f"buy_tag={row['buy']} sell_tag={row['sell']}"
        else:
            if not args.tag:
                raise ValueError("Either --tag or --from-daily-pick is required.")
            out_df, base = run(args.tag)
            tag_info = f"tag={args.tag}"

        # 出力先
        if not args.out:
            Path("data/analysis").mkdir(parents=True, exist_ok=True)
            out_path = Path(f"data/analysis/picks_{asof}.csv")
        else:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Wrote {out_path} ({len(out_df)} rows)")

        # ---------- ログ出力 ----------
        eligible = base.query("eligible==True").copy()
        rejected = base.query("eligible==False").copy()

        logger.info(f"ASOF={asof} {tag_info}")
        logger.info(
            "params: top_long=%d top_short=%d min_vol_ma=%.3f min_days=%s "
            "size_mode=%s capital=%.0f risk_pct=%.4f atr_window=%d atr_mult=%.2f "
            "vol_target_pct=%.4f lot=%d min_notional=%.0f max_notional=%.0f",
            args.top_long, args.top_short, args.min_vol_ma, str(args.min_days),
            args.size_mode, args.capital, args.risk_pct, args.atr_window, args.atr_mult,
            args.vol_target_pct, args.lot, args.min_notional, args.max_notional
        )

        logger.info("universe=%d  eligible=%d  rejected=%d", len(base), len(eligible), len(rejected))
        if not rejected.empty:
            top_reasons = (rejected["reason"].str.split(";").explode()
                           .value_counts().head(8).to_dict())
            logger.info("reject_reasons_top: %s", top_reasons)

        buy = out_df.query("side=='BUY'").copy()
        sell = out_df.query("side=='SELL'").copy()
        overlap = set(buy["ticker"]) & set(sell["ticker"])
        logger.info("BUY=%d  SELL=%d  overlap=%d", len(buy), len(sell), len(overlap))

        def head_rows(df: pd.DataFrame, k=15):
            cols = ["ticker", "score", "vol_maV", "close", "size", "notional"]
            return df[cols].head(k).to_dict("records")

        logger.info("BUY_top: %s", head_rows(buy.sort_values("score", ascending=False), 15))
        logger.info("SELL_top: %s", head_rows(sell.sort_values("score", ascending=True), 15))

        # ---------- 明細CSV（任意） ----------
        if args.explain_out:
            exp_path = Path(args.explain_out)
            exp_path.parent.mkdir(parents=True, exist_ok=True)
            # 見やすく：picked/side を付与して出す
            picked = out_df[["ticker", "side"]].copy()
            base2 = base.merge(picked, on="ticker", how="left")
            base2["picked"] = base2["side"].notna()
            base2.sort_values(["picked", "eligible", "score"], ascending=[False, False, False]) \
                 .to_csv(exp_path, index=False, encoding="utf-8-sig")
            logger.info("wrote explain csv: %s (rows=%d)", str(exp_path), len(base2))

    finally:
        con.close()


if __name__ == "__main__":
    main()
