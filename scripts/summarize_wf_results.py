"""
Summarize walk-forward results under data/analysis.

Inputs (per tag like T90_V30):
  - data/analysis/wf_results_<tag>.csv (train/test window metrics)
  - data/analysis/wf_daily_<tag>.csv   (daily returns for test windows, optional)

Outputs:
  - data/analysis/wf_summary_table.csv
  - data/analysis/wf_summary_bar.png
  - data/analysis/wf_summary_scatter.png

Exit codes:
  0: success
  2: no target files or input invalid
  4: unexpected exception
"""

import os
import sys
import re
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = os.path.join("data", "analysis")


def find_tags() -> Dict[str, Tuple[str, Optional[str]]]:
    """Return mapping tag -> (results_csv, daily_csv_or_None)."""
    if not os.path.isdir(OUTDIR):
        return {}
    rx = re.compile(r"^wf_results_(T\d+_V\d+)\.csv$")
    mapping: Dict[str, Tuple[str, Optional[str]]] = {}
    for name in os.listdir(OUTDIR):
        m = rx.match(name)
        if not m:
            continue
        tag = m.group(1)
        res_path = os.path.join(OUTDIR, name)
        daily_candidate = os.path.join(OUTDIR, f"wf_daily_{tag}.csv")
        mapping[tag] = (res_path, daily_candidate if os.path.exists(daily_candidate) else None)
    return mapping


def max_drawdown(cum_series: pd.Series) -> float:
    """Return max drawdown (min of cumulative/rolling_max - 1). <= 0."""
    if cum_series is None or len(cum_series) == 0:
        return float("nan")
    rollmax = cum_series.cummax()
    dd = cum_series / rollmax - 1.0
    return float(dd.min())


def metrics_for_tag(tag: str, res_csv: str, daily_csv: Optional[str]) -> Optional[dict]:
    try:
        df_res = pd.read_csv(res_csv)
    except Exception as e:
        print(f"[WARN] failed to read {res_csv}: {e}")
        return None
    if df_res.empty:
        print(f"[WARN] empty results: {res_csv}")
        return None

    # Basic metrics from window-level results
    n_windows = int(len(df_res))
    test_means = pd.to_numeric(df_res.get("test_mean"), errors="coerce")
    test_mean_avg = float(test_means.mean()) if n_windows > 0 else float("nan")
    test_mean_std = float(test_means.std(ddof=1)) if n_windows > 1 else 0.0
    win_rate = float((test_means > 0).mean()) if n_windows > 0 else float("nan")

    # Daily-based metrics
    cum_end = float("nan")
    mdd = float("nan")
    if daily_csv and os.path.exists(daily_csv):
        try:
            df_daily = pd.read_csv(daily_csv)
            if not df_daily.empty and "ret" in df_daily.columns:
                ret = pd.to_numeric(df_daily["ret"], errors="coerce").fillna(0.0)
                cum = (1.0 + ret).cumprod()
                if len(cum) > 0:
                    cum_end = float(cum.iloc[-1])
                    mdd = max_drawdown(cum)
        except Exception as e:
            print(f"[WARN] failed to read daily {daily_csv}: {e}")

    return {
        "tag": tag,
        "n_windows": n_windows,
        "test_mean_avg": test_mean_avg,
        "test_mean_std": test_mean_std,
        "win_rate": win_rate,
        "cum_return_end": cum_end,
        "max_drawdown": mdd,
    }


def sort_tag_key(tag: str) -> Tuple[int, int]:
    m = re.match(r"T(\d+)_V(\d+)", tag)
    if not m:
        return (999999, 999999)
    return (int(m.group(1)), int(m.group(2)))


def load_pairs() -> Dict[str, Tuple[str, Optional[str]]]:
    return find_tags()


def main() -> int:
    print(f"[INFO] scanning: {OUTDIR}")
    pairs = load_pairs()
    if not pairs:
        print(f"[ERROR] no wf_results_T*_V*.csv under {OUTDIR}")
        return 2

    rows: List[dict] = []
    for tag in sorted(pairs.keys(), key=sort_tag_key):
        res_csv, daily_csv = pairs[tag]
        print(f"[INFO] tag={tag} res='{os.path.basename(res_csv)}' daily='{os.path.basename(daily_csv) if daily_csv else 'N/A'}'")
        m = metrics_for_tag(tag, res_csv, daily_csv)
        if m is None:
            print(f"[WARN] skip tag={tag} due to invalid inputs")
            continue
        rows.append(m)

    if not rows:
        print("[ERROR] no valid tags to summarize")
        return 2

    os.makedirs(OUTDIR, exist_ok=True)
    df_sum = pd.DataFrame(rows)
    out_csv = os.path.join(OUTDIR, "wf_summary_table.csv")
    df_sum.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {out_csv} ({len(df_sum)} rows)")

    # Bar plot: tag vs cum_return_end
    plt.figure()
    df_bar = df_sum.dropna(subset=["cum_return_end"]) if "cum_return_end" in df_sum.columns else df_sum.iloc[0:0]
    if not df_bar.empty:
        plt.bar(df_bar["tag"], df_bar["cum_return_end"].astype(float))
        plt.xticks(rotation=45, ha="right")
    plt.title("Walk-Forward Cumulative Return (by config)")
    plt.xlabel("tag")
    plt.ylabel("cum_return_end")
    plt.tight_layout()
    out_bar = os.path.join(OUTDIR, "wf_summary_bar.png")
    plt.savefig(out_bar)
    print(f"[INFO] wrote {out_bar}")

    # Scatter plot: test_mean_avg vs max_drawdown, annotate by tag
    plt.figure()
    need_cols = ["test_mean_avg", "max_drawdown", "tag"]
    df_sc = df_sum.dropna(subset=["test_mean_avg", "max_drawdown"]) if all(c in df_sum.columns for c in need_cols) else df_sum.iloc[0:0]
    if not df_sc.empty:
        x = df_sc["test_mean_avg"].astype(float)
        y = df_sc["max_drawdown"].astype(float)
        plt.scatter(x, y)
        for _, r in df_sc.iterrows():
            plt.annotate(str(r["tag"]), (float(r["test_mean_avg"]), float(r["max_drawdown"])) , xytext=(3,3), textcoords="offset points")
    plt.title("Avg test mean vs Max drawdown (lower is better)")
    plt.xlabel("test_mean_avg")
    plt.ylabel("max_drawdown")
    plt.tight_layout()
    out_scat = os.path.join(OUTDIR, "wf_summary_scatter.png")
    plt.savefig(out_scat)
    print(f"[INFO] wrote {out_scat}")

    try:
        print("✅ Done.")
    except Exception:
        print("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] unexpected exception: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(4)
