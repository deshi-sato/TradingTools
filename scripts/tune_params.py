#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Search best parameters for generate_picks_from_daily.py to maximize BUY/SELL win rates
over a given date range by reproducing generate+eval logic in-memory (no CSV writes).

Parameters tuned:
- --buy-overbought
- --sell-oversold
- --upper-wick-ratio
- --lower-wick-ratio

Usage example:
  python scripts/tune_params.py \
    --db-path ./rss_daily.db \
    --start 2024-01-01 --end 2024-12-31 \
    --min-turnover 200000000 --topn 1 \
    --index-ticker 1306.T --disable-sell-in-uptrend

This prints the best combo and a short leaderboard.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

# Import helpers from generate_picks_from_daily
import sys
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_picks_from_daily import (
    read_daily,
    add_features,
    score_buy_row,
    score_sell_row,
    trend_up_idx,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune parameters for pick generation to maximize BUY/SELL win rate")
    p.add_argument("--db-path", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--min-turnover", type=float, default=200_000_000)
    p.add_argument("--topn", type=int, default=1)
    p.add_argument("--index-ticker", default="1306.T")
    p.add_argument("--disable-sell-in-uptrend", action="store_true")
    # grid options (coarse -> refine around best later if desired)
    p.add_argument("--grid-buy", default="20,30,40,50,60", help="Comma list for buy-overbought candidates")
    p.add_argument("--grid-sell", default="70,80,90", help="Comma list for sell-oversold candidates")
    p.add_argument("--grid-uw", default="0.8,1.0,1.5", help="Comma list for upper-wick-ratio candidates")
    p.add_argument("--grid-lw", default="0.8,1.0,1.5", help="Comma list for lower-wick-ratio candidates")
    return p.parse_args()


@dataclass
class Stats:
    buy_n: int = 0
    sell_n: int = 0
    buy_wins: int = 0
    sell_wins: int = 0

    @property
    def buy_rate(self) -> float:
        return (self.buy_wins / self.buy_n) if self.buy_n else 0.0

    @property
    def sell_rate(self) -> float:
        return (self.sell_wins / self.sell_n) if self.sell_n else 0.0

    @property
    def avg_rate(self) -> float:
        # simple average of the two win rates
        return 0.5 * (self.buy_rate + self.sell_rate)


def compute_features(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    start_buf = start - timedelta(days=150)
    df = df[(df["date"] >= pd.Timestamp(start_buf)) & (df["date"] <= pd.Timestamp(end))].copy()
    # feature engineering per generate_picks_from_daily
    return pd.concat((add_features(g) for _, g in df.groupby("ticker", sort=False)), ignore_index=True)


def evaluate_combo(
    df_feat: pd.DataFrame,
    start: date,
    end: date,
    min_turnover: float,
    topn: int,
    index_ticker: Optional[str],
    disable_sell_in_uptrend: bool,
    buy_overbought: float,
    sell_oversold: float,
    upper_wick_ratio_thr: float,
    lower_wick_ratio_thr: float,
) -> Stats:
    # precompute rolling/prev columns
    df = df_feat.copy()
    # scores for the current day (we use prev=shift(1) below)
    buy_today = pd.concat((g.apply(score_buy_row, axis=1) for _, g in df.groupby("ticker", sort=False)))
    sell_today = pd.concat((g.apply(score_sell_row, axis=1) for _, g in df.groupby("ticker", sort=False)))
    buy_today.index = df.index
    sell_today.index = df.index

    # previous-day features and scores
    df["score_buy_prev"] = buy_today.groupby(df["ticker"]).shift(1)
    df["score_sell_prev"] = sell_today.groupby(df["ticker"]).shift(1)
    df["turnover_prev"] = df.groupby("ticker")["turnover"].shift(1)
    df["rsi3_prev"] = df.groupby("ticker")["rsi3"].shift(1)
    df["upper_wick_prev"] = df.groupby("ticker")["upper_wick_ratio"].shift(1)
    df["lower_wick_prev"] = df.groupby("ticker")["lower_wick_ratio"].shift(1)
    df["pos_prev"] = df.groupby("ticker")["pos_in_range"].shift(1)

    # date ladder and next-day mapping
    days = sorted(df["date"].dropna().unique())
    next_map: Dict[pd.Timestamp, Optional[pd.Timestamp]] = {
        days[i]: (days[i + 1] if i + 1 < len(days) else None) for i in range(len(days))
    }

    # index slice for trend gating
    idx_df = None
    if index_ticker:
        idx_df = df[df["ticker"] == index_ticker].loc[:, ["date", "close", "ma5", "ma25", "ma75"]].copy()

    stats = Stats()

    for d, daydf in df.groupby("date"):
        next_d = next_map.get(d)
        if next_d is None:
            continue
        # Only evaluate picks whose trade date (next_d) is inside target window
        if not (pd.Timestamp(start) <= pd.Timestamp(next_d) <= pd.Timestamp(end)):
            continue

        allow_sell = True
        if idx_df is not None and disable_sell_in_uptrend:
            idx_day = idx_df[idx_df["date"] == d]
            if not idx_day.empty and trend_up_idx(idx_day):
                allow_sell = False

        cands = daydf.dropna(
            subset=[
                "score_buy_prev",
                "score_sell_prev",
                "turnover_prev",
                "rsi3_prev",
                "upper_wick_prev",
                "lower_wick_prev",
                "pos_prev",
            ]
        ).copy()
        cands = cands[cands["turnover_prev"] >= float(min_turnover)]

        buy_mask = (
            (cands["rsi3_prev"] <= buy_overbought)
            & (cands["upper_wick_prev"] <= upper_wick_ratio_thr)
            & (cands["pos_prev"] <= 0.98)
        )
        sell_mask = (
            (cands["rsi3_prev"] >= sell_oversold)
            & (cands["lower_wick_prev"] <= lower_wick_ratio_thr)
            & (cands["pos_prev"] >= 0.02)
        )

        # Select topn by score then turnover
        # BUY
        buy_picks = []
        if buy_mask.any():
            buy_cands = cands[buy_mask].sort_values(
                ["score_buy_prev", "turnover_prev"], ascending=[False, False]
            ).head(topn)
            for _, r in buy_cands.iterrows():
                if r["score_buy_prev"] > 0:
                    buy_picks.append(str(r["ticker"]))

        # SELL
        sell_picks = []
        if allow_sell and sell_mask.any():
            sell_cands = cands[sell_mask].sort_values(
                ["score_sell_prev", "turnover_prev"], ascending=[False, False]
            ).head(topn)
            for _, r in sell_cands.iterrows():
                if r["score_sell_prev"] > 0:
                    sell_picks.append(str(r["ticker"]))

        if not buy_picks and not sell_picks:
            continue

        # Compute next-day open/close from df for returns (Open->Close)
        next_rows = df[df["date"] == next_d].set_index("ticker")[
            ["open", "close"]
        ]

        for code in buy_picks[:topn]:
            if code in next_rows.index:
                o = float(next_rows.loc[code, "open"])
                c = float(next_rows.loc[code, "close"])
                if o > 0 and c > 0:
                    stats.buy_n += 1
                    stats.buy_wins += int((c - o) > 0)

        for code in sell_picks[:topn]:
            if code in next_rows.index:
                o = float(next_rows.loc[code, "open"])
                c = float(next_rows.loc[code, "close"])
                if o > 0 and c > 0:
                    stats.sell_n += 1
                    stats.sell_wins += int((o - c) > 0)

    return stats


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    print(f"[INFO] Loading daily bars from {args.db_path} ...")
    df = read_daily(args.db_path)
    print(f"[INFO] Rows: {len(df):,}, tickers: {df['ticker'].nunique():,}, dates: {df['date'].nunique():,}")

    print("[INFO] Computing features ...")
    df_feat = compute_features(df, start, end)

    grid_buy = [float(x) for x in str(args.grid_buy).split(',') if str(x).strip()]
    grid_sell = [float(x) for x in str(args.grid_sell).split(',') if str(x).strip()]
    grid_uw = [float(x) for x in str(args.grid_uw).split(',') if str(x).strip()]
    grid_lw = [float(x) for x in str(args.grid_lw).split(',') if str(x).strip()]

    combos: List[Tuple[float, float, float, float]] = []
    for bo in grid_buy:
        for so in grid_sell:
            for uw in grid_uw:
                for lw in grid_lw:
                    combos.append((bo, so, uw, lw))

    print(f"[INFO] Searching {len(combos)} combos ...")
    results: List[Tuple[Tuple[float, float, float, float], Stats]] = []

    for i, (bo, so, uw, lw) in enumerate(combos, 1):
        stats = evaluate_combo(
            df_feat=df_feat,
            start=start,
            end=end,
            min_turnover=args.min_turnover,
            topn=args.topn,
            index_ticker=(None if str(args.index_ticker).strip().lower() in ["", "none"] else args.index_ticker),
            disable_sell_in_uptrend=args.disable_sell_in_uptrend,
            buy_overbought=bo,
            sell_oversold=so,
            upper_wick_ratio_thr=uw,
            lower_wick_ratio_thr=lw,
        )
        results.append(((bo, so, uw, lw), stats))
        if i % 10 == 0:
            print(
                f"[PROGRESS] {i}/{len(combos)}: bo={bo}, so={so}, uw={uw}, lw={lw} | "
                f"BUY {stats.buy_rate:.1%} ({stats.buy_wins}/{stats.buy_n}), "
                f"SELL {stats.sell_rate:.1%} ({stats.sell_wins}/{stats.sell_n})"
            )

    # sort by average win rate, then by total picks (to break ties)
    results.sort(key=lambda x: (x[1].avg_rate, x[1].buy_n + x[1].sell_n), reverse=True)

    print("\n=== Top 5 combos by average win rate ===")
    for (bo, so, uw, lw), st in results[:5]:
        print(
            f"bo={bo:.1f}, so={so:.1f}, uw={uw:.2f}, lw={lw:.2f} | "
            f"BUY {st.buy_rate:.1%} ({st.buy_wins}/{st.buy_n}), "
            f"SELL {st.sell_rate:.1%} ({st.sell_wins}/{st.sell_n}), avg={st.avg_rate:.1%}"
        )

    best, st = results[0]
    bo, so, uw, lw = best
    print("\n=== Best parameters ===")
    print(
        f"--buy-overbought {bo:.1f} --sell-oversold {so:.1f} "
        f"--upper-wick-ratio {uw:.2f} --lower-wick-ratio {lw:.2f}"
    )
    print(
        f"BUY {st.buy_rate:.2%} ({st.buy_wins}/{st.buy_n}), "
        f"SELL {st.sell_rate:.2%} ({st.sell_wins}/{st.sell_n}), avg={st.avg_rate:.2%}"
    )


if __name__ == "__main__":
    main()

