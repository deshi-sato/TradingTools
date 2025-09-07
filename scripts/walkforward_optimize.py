#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monthly walk-forward optimization for generate_picks_from_daily parameters.

For each month in [start, end], it:
- Optimizes parameters on a trailing training window (in months)
- Evaluates the chosen parameters on the next month (out-of-sample)
- Aggregates results across months to propose robust parameters

Outputs CSV reports under ./reports/:
- walkforward_folds.csv: per-month OOS metrics and chosen parameters
- walkforward_summary.csv: top parameter combos by average OOS win rate and stability

Note: Uses in-memory evaluation consistent with scripts/tune_params.py and
scripts/generate_picks_from_daily.py logic (Open->Close win-rate).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, date, timedelta
import calendar

import pandas as pd

import sys
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_picks_from_daily import read_daily, add_features, trend_up_idx  # noqa: E402
from tune_params import evaluate_combo  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward optimization for pick parameters")
    p.add_argument("--db-path", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD (validation start)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (validation end)")
    p.add_argument("--train-months", type=int, default=6, help="Training window length (months)")
    p.add_argument("--min-turnover", type=float, default=200_000_000)
    p.add_argument("--topn", type=int, default=1)
    p.add_argument("--index-ticker", default="1306.T")
    p.add_argument("--disable-sell-in-uptrend", action="store_true")
    # grids
    p.add_argument("--grid-buy", default="20,30,40,50,60")
    p.add_argument("--grid-sell", default="70,80,90")
    p.add_argument("--grid-uw", default="0.8,1.0,1.5")
    p.add_argument("--grid-lw", default="0.8,1.0,1.5")
    return p.parse_args()


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def month_end(d: date) -> date:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, last_day)


def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


@dataclass
class FoldResult:
    val_month: str
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    buy_overbought: float
    sell_oversold: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    buy_wins: int
    buy_n: int
    sell_wins: int
    sell_n: int

    @property
    def buy_rate(self) -> float:
        return (self.buy_wins / self.buy_n) if self.buy_n else 0.0

    @property
    def sell_rate(self) -> float:
        return (self.sell_wins / self.sell_n) if self.sell_n else 0.0

    @property
    def avg_rate(self) -> float:
        return 0.5 * (self.buy_rate + self.sell_rate)


def compute_features_full(df: pd.DataFrame) -> pd.DataFrame:
    # Compute features for entire dataset per ticker, then we slice by dates per fold.
    return pd.concat((add_features(g) for _, g in df.groupby("ticker", sort=False)), ignore_index=True)


def main() -> None:
    args = parse_args()
    val_start = datetime.strptime(args.start, "%Y-%m-%d").date()
    val_end = datetime.strptime(args.end, "%Y-%m-%d").date()

    print(f"[INFO] Loading daily bars from {args.db_path} ...")
    df = read_daily(args.db_path)
    print(f"[INFO] Rows: {len(df):,}, tickers: {df['ticker'].nunique():,}, dates: {df['date'].nunique():,}")

    print("[INFO] Computing features across entire dataset ...")
    df_feat_all = compute_features_full(df)

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
    print(f"[INFO] Grid size: {len(combos)}")

    # iterate monthly folds
    folds: List[FoldResult] = []

    cur = month_start(val_start)
    end_m = month_end(val_end)
    while cur <= end_m:
        v_start = month_start(cur)
        v_end = month_end(cur)
        # training window
        t_end = v_start - timedelta(days=1)
        t_start = month_start(add_months(v_start, -int(args.train_months)))

        # skip if training ends before any data present
        # (We will still try; evaluate_combo gracefully handles no picks.)
        best_combo = None
        best_metric = -1.0

        for (bo, so, uw, lw) in combos:
            stats_tr = evaluate_combo(
                df_feat=df_feat_all,
                start=t_start,
                end=t_end,
                min_turnover=args.min_turnover,
                topn=args.topn,
                index_ticker=(None if str(args.index_ticker).strip().lower() in ["", "none"] else args.index_ticker),
                disable_sell_in_uptrend=args.disable_sell_in_uptrend,
                buy_overbought=bo,
                sell_oversold=so,
                upper_wick_ratio_thr=uw,
                lower_wick_ratio_thr=lw,
            )
            metric = stats_tr.avg_rate
            if metric > best_metric:
                best_metric = metric
                best_combo = (bo, so, uw, lw)

        # evaluate on validation month
        bo, so, uw, lw = best_combo if best_combo else (grid_buy[0], grid_sell[0], grid_uw[0], grid_lw[0])
        stats_v = evaluate_combo(
            df_feat=df_feat_all,
            start=v_start,
            end=v_end,
            min_turnover=args.min_turnover,
            topn=args.topn,
            index_ticker=(None if str(args.index_ticker).strip().lower() in ["", "none"] else args.index_ticker),
            disable_sell_in_uptrend=args.disable_sell_in_uptrend,
            buy_overbought=bo,
            sell_oversold=so,
            upper_wick_ratio_thr=uw,
            lower_wick_ratio_thr=lw,
        )

        folds.append(
            FoldResult(
                val_month=f"{v_start:%Y-%m}",
                train_start=t_start,
                train_end=t_end,
                val_start=v_start,
                val_end=v_end,
                buy_overbought=bo,
                sell_oversold=so,
                upper_wick_ratio=uw,
                lower_wick_ratio=lw,
                buy_wins=stats_v.buy_wins,
                buy_n=stats_v.buy_n,
                sell_wins=stats_v.sell_wins,
                sell_n=stats_v.sell_n,
            )
        )

        print(
            f"[FOLD {v_start:%Y-%m}] OOS BUY {folds[-1].buy_rate:.1%} ({folds[-1].buy_wins}/{folds[-1].buy_n}), "
            f"SELL {folds[-1].sell_rate:.1%} ({folds[-1].sell_wins}/{folds[-1].sell_n}) | "
            f"params: bo={bo}, so={so}, uw={uw}, lw={lw}"
        )

        # advance to next month
        cur = month_start(add_months(cur, 1))

    # aggregate per-parameter combo across folds
    from collections import defaultdict

    agg: Dict[Tuple[float, float, float, float], Dict[str, float]] = defaultdict(lambda: {
        "buy_wins": 0.0,
        "buy_n": 0.0,
        "sell_wins": 0.0,
        "sell_n": 0.0,
        "folds": 0.0,
    })

    for fr in folds:
        key = (fr.buy_overbought, fr.sell_oversold, fr.upper_wick_ratio, fr.lower_wick_ratio)
        a = agg[key]
        a["buy_wins"] += fr.buy_wins
        a["buy_n"] += fr.buy_n
        a["sell_wins"] += fr.sell_wins
        a["sell_n"] += fr.sell_n
        a["folds"] += 1

    rows = []
    for (bo, so, uw, lw), a in agg.items():
        buy_rate = (a["buy_wins"] / a["buy_n"]) if a["buy_n"] else 0.0
        sell_rate = (a["sell_wins"] / a["sell_n"]) if a["sell_n"] else 0.0
        avg_rate = 0.5 * (buy_rate + sell_rate)
        rows.append({
            "buy_overbought": bo,
            "sell_oversold": so,
            "upper_wick_ratio": uw,
            "lower_wick_ratio": lw,
            "folds": int(a["folds"]),
            "buy_rate": buy_rate,
            "sell_rate": sell_rate,
            "avg_rate": avg_rate,
        })

    summary = pd.DataFrame(rows).sort_values(["avg_rate", "folds"], ascending=[False, False])

    # outputs
    reports = Path("./reports")
    reports.mkdir(exist_ok=True)

    folds_df = pd.DataFrame([
        {
            "val_month": fr.val_month,
            "train_start": fr.train_start.strftime("%Y-%m-%d"),
            "train_end": fr.train_end.strftime("%Y-%m-%d"),
            "val_start": fr.val_start.strftime("%Y-%m-%d"),
            "val_end": fr.val_end.strftime("%Y-%m-%d"),
            "buy_overbought": fr.buy_overbought,
            "sell_oversold": fr.sell_oversold,
            "upper_wick_ratio": fr.upper_wick_ratio,
            "lower_wick_ratio": fr.lower_wick_ratio,
            "buy_wins": fr.buy_wins,
            "buy_n": fr.buy_n,
            "sell_wins": fr.sell_wins,
            "sell_n": fr.sell_n,
            "buy_rate": fr.buy_rate,
            "sell_rate": fr.sell_rate,
            "avg_rate": fr.avg_rate,
        }
        for fr in folds
    ])
    folds_csv = reports / "walkforward_folds.csv"
    folds_df.to_csv(folds_csv, index=False, encoding="utf-8-sig")

    summary_csv = reports / "walkforward_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n=== Walk-forward OOS summary (top 10) ===")
    for _, r in summary.head(10).iterrows():
        print(
            f"bo={r['buy_overbought']}, so={r['sell_oversold']}, uw={r['upper_wick_ratio']}, lw={r['lower_wick_ratio']} | "
            f"folds={int(r['folds'])}, BUY {r['buy_rate']:.1%}, SELL {r['sell_rate']:.1%}, avg={r['avg_rate']:.1%}"
        )

    print(f"\nSaved folds: {folds_csv.resolve()}\nSaved summary: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()

