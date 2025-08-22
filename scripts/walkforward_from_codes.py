"""
Walk-forward evaluator for daily code scores.
WF_TRAIN / WF_TEST control train/test window lengths (days).
Input:  data/score_daily.codes.csv with columns: date, code, score, next_return
Outputs under data/analysis: wf_results.csv, wf_daily.csv, wf_cum.png, wf_window_bars.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CODES_CSV = r"data/score_daily.codes.csv"
OUTDIR = r"data/analysis"

TRAIN_DAYS = int(os.getenv("WF_TRAIN", "180"))
TEST_DAYS = int(os.getenv("WF_TEST", "30"))
N_LIST = [5, 10, 20]

os.makedirs(OUTDIR, exist_ok=True)


def daily_mean_ret(sub: pd.DataFrame, top: bool, n: int) -> float:
    """Mean next_return for Top/Bottom N by score on a single day."""
    if top:
        part = sub.sort_values("score", ascending=False).head(n)
    else:
        part = sub.sort_values("score", ascending=True).head(n)
    return float(part["next_return"].mean()) if len(part) > 0 else np.nan


def window_eval(df: pd.DataFrame, train_dates: pd.Series) -> dict:
    """Pick best of candidate strategies on train period (by mean return)."""
    train_df = df[df["date"].isin(set(train_dates))]
    if train_df.empty:
        return {"side": "Top", "n": 5, "train_mean": np.nan}
    by_day = train_df.groupby("date", sort=True)
    rows = []
    for n in N_LIST:
        top_series = by_day.apply(lambda g: daily_mean_ret(g, top=True, n=n))
        bot_series = by_day.apply(lambda g: daily_mean_ret(g, top=False, n=n))
        rows.append(("Top", n, float(top_series.mean())))
        rows.append(("Bottom", n, float(bot_series.mean())))
    best = max(rows, key=lambda x: x[2] if not np.isnan(x[2]) else -1e18)
    return {"side": best[0], "n": best[1], "train_mean": best[2]}


def apply_strategy(df: pd.DataFrame, test_dates: pd.Series, side: str, n: int) -> pd.DataFrame:
    """Apply chosen strategy on test period and return daily returns."""
    test_df = df[df["date"].isin(set(test_dates))]
    out = []
    for d, g in test_df.groupby("date", sort=True):
        r = daily_mean_ret(g, top=(side == "Top"), n=n)
        if not np.isnan(r):
            out.append({"date": d, "ret": float(r)})
    return pd.DataFrame(out).sort_values("date")


def main() -> int:
    # Input validation
    if not os.path.exists(CODES_CSV):
        print(f"[ERROR] not found: {CODES_CSV}", file=sys.stderr)
        return 1
    df = pd.read_csv(CODES_CSV)
    required = ["date", "code", "score", "next_return"]
    for c in required:
        if c not in df.columns:
            print(f"[ERROR] missing column '{c}' in {CODES_CSV}", file=sys.stderr)
            return 1
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)

    days = pd.Series(sorted(df["date"].unique()))
    if len(days) < (TRAIN_DAYS + TEST_DAYS):
        print(f"[ERROR] insufficient days: {len(days)} < {TRAIN_DAYS + TEST_DAYS}", file=sys.stderr)
        return 3

    print(f"[INFO] settings: TRAIN_DAYS={TRAIN_DAYS} TEST_DAYS={TEST_DAYS}")
    print(f"[INFO] available days: {len(days)} (require >= {TRAIN_DAYS + TEST_DAYS})")

    results = []
    wf_daily_all = []

    i = 0
    while i + TRAIN_DAYS + TEST_DAYS <= len(days):
        train_dates = days.iloc[i : i + TRAIN_DAYS]
        test_dates = days.iloc[i + TRAIN_DAYS : i + TRAIN_DAYS + TEST_DAYS]

        sel = window_eval(df, train_dates)
        side, n, train_mean = sel["side"], sel["n"], sel["train_mean"]

        d_train_start, d_train_end = train_dates.iloc[0], train_dates.iloc[-1]
        d_test_start, d_test_end = test_dates.iloc[0], test_dates.iloc[-1]

        test_daily = apply_strategy(df, test_dates, side, n)
        test_mean = float(test_daily["ret"].mean()) if not test_daily.empty else np.nan

        results.append(
            {
                "train_start": d_train_start,
                "train_end": d_train_end,
                "test_start": d_test_start,
                "test_end": d_test_end,
                "chosen_side": side,
                "chosen_n": n,
                "train_mean": train_mean,
                "test_mean": test_mean,
            }
        )
        win_idx = len(results)
        print(
            f"[INFO] window#{win_idx}: "
            f"train {d_train_start}..{d_train_end}, "
            f"test {d_test_start}..{d_test_end}, "
            f"chosen {side}/N={n}, "
            f"train_mean={train_mean:.4f}, test_mean={'nan' if np.isnan(test_mean) else f'{test_mean:.4f}'}"
        )
        if not test_daily.empty:
            test_daily["window_id"] = win_idx
            wf_daily_all.append(test_daily)

        i += TEST_DAYS  # move by test window length

    # Outputs
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTDIR, "wf_results.csv"), index=False, encoding="utf-8-sig")

    wf_daily = (
        pd.concat(wf_daily_all, ignore_index=True)
        if wf_daily_all
        else pd.DataFrame(columns=["date", "ret", "window_id"])
    )
    wf_daily = wf_daily.sort_values(["date", "window_id"]) if not wf_daily.empty else wf_daily
    wf_daily.to_csv(os.path.join(OUTDIR, "wf_daily.csv"), index=False, encoding="utf-8-sig")

    # Plot: cumulative
    plt.figure()
    if not wf_daily.empty:
        s = (1.0 + wf_daily.set_index("date")["ret"].fillna(0)).cumprod()
        s.plot()
    plt.title("Walk-Forward Cumulative Return (Test periods only)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "wf_cum.png"))

    # Plot: window bars
    plt.figure()
    if not res_df.empty:
        x = range(1, len(res_df) + 1)
        y = res_df["test_mean"].fillna(0).values
        plt.bar(x, y)
        plt.xticks(x, [f"{i}" for i in x])
    plt.title("Walk-Forward Test Mean Return by Window")
    plt.xlabel("Window #")
    plt.ylabel("Mean next-day return")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "wf_window_bars.png"))

    try:
        print("✅ Done.")
    except Exception:
        # Fallback for consoles that cannot render emoji
        print("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] unexpected exception: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(4)
