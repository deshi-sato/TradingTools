"""
Pick top WF configs from wf_summary_table.csv with filters and sorting.

Env vars (defaults):
  WF_MIN_WINRATE = 0.70   # keep configs with win_rate >= this
  WF_MAX_DD      = -0.15  # keep configs with max_drawdown >= this (e.g., -0.15)
  WF_MIN_CUM     = 1.10   # keep configs with cum_return_end >= this
  WF_TOPN        = 3      # number of rows to output

Inputs:
  data/analysis/wf_summary_table.csv

Outputs:
  data/analysis/wf_summary_top.csv
  data/analysis/wf_summary_top_bar.png (x: tag, y: cum_return_end)

Exit codes:
  0: success
  2: wf_summary_table.csv not found / invalid
  3: no rows matched (empty outputs still written)
  4: unexpected exception
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = os.path.join("data", "analysis")
TABLE = os.path.join(OUTDIR, "wf_summary_table.csv")
TOP_CSV = os.path.join(OUTDIR, "wf_summary_top.csv")
TOP_BAR = os.path.join(OUTDIR, "wf_summary_top_bar.png")


def read_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


def read_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)


def main() -> int:
    min_win = read_env_float("WF_MIN_WINRATE", 0.70)
    max_dd = read_env_float("WF_MAX_DD", -0.15)
    min_cum = read_env_float("WF_MIN_CUM", 1.10)
    topn = read_env_int("WF_TOPN", 3)

    print(f"[INFO] filters: win_rate>={min_win}, max_drawdown>={max_dd}, cum_return_end>={min_cum}, topN={topn}")
    print(f"[INFO] input: {TABLE}")
    print(f"[INFO] outputs: {TOP_CSV}, {TOP_BAR}")

    if not os.path.exists(TABLE):
        print(f"[ERROR] not found: {TABLE}")
        return 2

    try:
        df = pd.read_csv(TABLE)
    except Exception as e:
        print(f"[ERROR] failed to read {TABLE}: {e}")
        return 2

    required = ["tag", "win_rate", "cum_return_end", "max_drawdown", "test_mean_std"]
    for c in required:
        if c not in df.columns:
            print(f"[ERROR] missing column '{c}' in {TABLE}")
            return 2

    # Ensure numeric
    df["win_rate"] = pd.to_numeric(df["win_rate"], errors="coerce")
    df["cum_return_end"] = pd.to_numeric(df["cum_return_end"], errors="coerce")
    df["max_drawdown"] = pd.to_numeric(df["max_drawdown"], errors="coerce")
    df["test_mean_std"] = pd.to_numeric(df["test_mean_std"], errors="coerce")

    # Filter
    cond = (
        (df["win_rate"] >= min_win)
        & (df["max_drawdown"] >= max_dd)
        & (df["cum_return_end"] >= min_cum)
    )
    picked = df.loc[cond].copy()
    print(f"[INFO] matched rows: {len(picked)} / {len(df)}")

    # Sort: cum_return_end desc, win_rate desc, test_mean_std asc (stability)
    if not picked.empty:
        picked = picked.sort_values(
            by=["cum_return_end", "win_rate", "test_mean_std"],
            ascending=[False, False, True],
            kind="mergesort",
        )

    out = picked.head(topn)
    os.makedirs(OUTDIR, exist_ok=True)
    out.to_csv(TOP_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {TOP_CSV} ({len(out)} rows)")

    # Bar plot (may be empty)
    plt.figure()
    if not out.empty:
        plt.bar(out["tag"], out["cum_return_end"].astype(float))
        plt.xticks(rotation=45, ha="right")
    plt.title("Top configs by cumulative return (filtered)")
    plt.xlabel("tag")
    plt.ylabel("cum_return_end")
    plt.tight_layout()
    plt.savefig(TOP_BAR)
    print(f"[INFO] wrote {TOP_BAR}")

    if picked.empty:
        print("[WARN] no configs matched filters")
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] unexpected exception: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(4)

