"""
Compare walk-forward daily results for selected tags (e.g., T60_V30,T90_V30).

Env:
  WF_COMPARE_TAGS: comma-separated tags like "T60_V30,T90_V30" (default: auto-pick 2 by availability)
  WF_MONTHS: number of recent months to include in monthly tables/plots (default: 6)

Inputs:
  data/analysis/wf_daily_<tag>.csv  (columns: date, ret, window_id)

Outputs under data/analysis:
  compare_summary.csv            (tag,days,mean,std,sharpe,cum_end,max_dd)
  compare_monthly_table.csv      (month,tag,days,win_rate,mean_ret)
  compare_cum.png                (overlay cumulative curves)
  compare_hist_<tag>.png         (return histogram per tag)
  compare_monthly_winrate.png    (grouped bar: monthly win rate)
  compare_monthly_mean.png       (grouped bar: monthly mean return)

Exit codes:
  0: success
  2: insufficient inputs (tags/daily files)
  4: unexpected exception
"""

import os
import sys
import math
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = os.path.join("data", "analysis")


def read_env_list(name: str) -> List[str]:
    v = os.getenv(name, "").strip()
    if not v:
        return []
    return [s.strip() for s in v.split(",") if s.strip()]


def read_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)


def find_available_tags() -> List[str]:
    if not os.path.isdir(OUTDIR):
        return []
    tags = []
    for name in os.listdir(OUTDIR):
        if name.startswith("wf_daily_T") and name.endswith(".csv"):
            tag = name[len("wf_daily_") : -len(".csv")]
            tags.append(tag)
    tags = sorted(set(tags))
    return tags


def _safe_tag(tag: str) -> str:
    # Keep alnum, underscore and dash; replace others with underscore
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in str(tag))


def pick_tags() -> List[str]:
    tags = read_env_list("WF_COMPARE_TAGS")
    if not tags:
        # Prefer standard set if available; include T180_V30 too
        avail = find_available_tags()
        preferred = ["T60_V30", "T90_V30", "T180_V30"]
        tags = [t for t in preferred if t in avail]
        # Fallback: add other available tags to reach at least 2
        if len(tags) < 2:
            for t in avail:
                if t not in tags:
                    tags.append(t)
                if len(tags) >= 2:
                    break
        if len(tags) < 2:
            return []
    # Validate files exist
    exist_tags = []
    for t in tags:
        if os.path.exists(os.path.join(OUTDIR, f"wf_daily_{t}.csv")):
            exist_tags.append(t)
        else:
            print(f"[WARN] missing daily file for tag={t}")
    # Need at least two
    if len(exist_tags) < 2:
        return []
    # Allow 2 or more tags (plots below handle N>=2)
    return exist_tags


def max_drawdown(cum_series: pd.Series) -> float:
    if cum_series is None or len(cum_series) == 0:
        return float("nan")
    rollmax = cum_series.cummax()
    dd = cum_series / rollmax - 1.0
    return float(dd.min())


def load_daily(tag: str) -> pd.DataFrame:
    path = os.path.join(OUTDIR, f"wf_daily_{tag}.csv")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_cum(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod()


def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "days", "win_rate", "mean_ret"]) 
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.to_period("M").astype(str)
    g = tmp.groupby("month", sort=True)
    out = g.apply(
        lambda gdf: pd.Series(
            {
                "days": int(gdf["ret"].count()),
                "win_rate": float((gdf["ret"] > 0).mean()),
                "mean_ret": float(gdf["ret"].mean()),
            }
        )
    ).reset_index()
    # month is already YYYY-MM string
    return out


def main() -> int:
    tags = pick_tags()
    months = read_env_int("WF_MONTHS", 6)

    print(f"[INFO] tags={tags}, months={months}")
    if len(tags) < 2:
        print(f"[ERROR] need at least 2 comparable tags with daily files under {OUTDIR}")
        return 2

    # Load daily for tags
    dailies: Dict[str, pd.DataFrame] = {}
    for t in tags:
        try:
            d = load_daily(t)
        except Exception as e:
            print(f"[ERROR] failed to read daily for {t}: {e}")
            return 2
        dailies[t] = d

    # Summary metrics
    rows = []
    for t, df in dailies.items():
        ret = pd.to_numeric(df.get("ret"), errors="coerce").fillna(0.0)
        days = int(ret.count())
        mean = float(ret.mean()) if days > 0 else float("nan")
        std = float(ret.std(ddof=1)) if days > 1 else 0.0
        sharpe = float(mean / std * math.sqrt(252)) if std > 0 else float("nan")
        cum = make_cum(ret)
        cum_end = float(cum.iloc[-1]) if days > 0 else float("nan")
        mdd = max_drawdown(cum)
        # Max consecutive losses (ret < 0)
        max_consec_losses = 0
        cur = 0
        for x in ret.tolist():
            if x < 0:
                cur += 1
                if cur > max_consec_losses:
                    max_consec_losses = cur
            else:
                cur = 0
        # Profit Factor (PF): gross profit / gross loss (absolute)
        gross_profit = float(ret[ret > 0].sum()) if days > 0 else float("nan")
        gross_loss = float(ret[ret < 0].sum()) if days > 0 else float("nan")  # negative or 0
        if days == 0 or np.isnan(gross_profit) or np.isnan(gross_loss):
            pf = float("nan")
        else:
            if gross_loss < 0:
                pf = gross_profit / abs(gross_loss)
            else:
                pf = float("inf") if gross_profit > 0 else float("nan")
        rows.append({
            "tag": t,
            "days": days,
            "mean": mean,
            "std": std,
            "sharpe": sharpe,
            "cum_end": cum_end,
            "max_dd": mdd,
            "max_consec_losses": int(max_consec_losses),
            "pf": pf,
        })
    df_sum = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    df_sum.to_csv(os.path.join(OUTDIR, "compare_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote compare_summary.csv ({len(df_sum)} rows)")

    # Align dates for cumulative plot
    plt.figure()
    for t in tags:
        df = dailies[t]
        if df.empty:
            continue
        ret = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)
        cum = make_cum(ret)
        x = pd.to_datetime(df["date"]).values
        plt.plot(x, cum, label=t)
    plt.title("WF Compare: Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "compare_cum.png"))
    print("[INFO] wrote compare_cum.png")

    # Histograms per tag
    for t in tags:
        df = dailies[t]
        plt.figure()
        if not df.empty:
            ret = pd.to_numeric(df["ret"], errors="coerce").dropna()
            plt.hist(ret, bins=50)
        plt.title(f"Return histogram - {t}")
        plt.xlabel("daily return")
        plt.ylabel("count")
        plt.tight_layout()
        fn = os.path.join(OUTDIR, f"compare_hist_{_safe_tag(t)}.png")
        plt.savefig(fn)
        print(f"[INFO] wrote {os.path.basename(fn)}")

    # Monthly aggregation and limit to recent N months
    monthly_list = []
    for t in tags:
        m = monthly_agg(dailies[t])
        m["tag"] = t
        monthly_list.append(m)
    df_month = pd.concat(monthly_list, ignore_index=True) if monthly_list else pd.DataFrame()
    if not df_month.empty:
        # Take last N months by month string
        months_sorted = sorted(df_month["month"].unique())
        months_keep = months_sorted[-months:] if months > 0 else months_sorted
        df_month = df_month[df_month["month"].isin(months_keep)]
    df_month.to_csv(os.path.join(OUTDIR, "compare_monthly_table.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote compare_monthly_table.csv ({len(df_month)} rows)")

    # Monthly win rate plot
    plt.figure()
    if not df_month.empty:
        months_x = sorted(df_month["month"].unique())
        x_idx = np.arange(len(months_x))
        n = max(2, len(tags))
        width = min(0.8 / n, 0.35)

        def first_or_zero(month_val: str, tag_val: str, col: str) -> float:
            ser = df_month.loc[(df_month["month"] == month_val) & (df_month["tag"] == tag_val), col]
            if ser.empty:
                return 0.0
            try:
                return float(ser.iloc[0])
            except Exception:
                return 0.0

        for i, t in enumerate(tags):
            offset = (i - (n - 1) / 2) * width
            y = [first_or_zero(m, t, "win_rate") for m in months_x]
            plt.bar(x_idx + offset, y, width, label=t)

        plt.xticks(x_idx, months_x, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.legend()
    plt.title("Monthly win rate by tag")
    plt.xlabel("month")
    plt.ylabel("win_rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "compare_monthly_winrate.png"))
    print("[INFO] wrote compare_monthly_winrate.png")

    # Monthly mean plot
    plt.figure()
    if not df_month.empty:
        months_x = sorted(df_month["month"].unique())
        x_idx = np.arange(len(months_x))
        n = max(2, len(tags))
        width = min(0.8 / n, 0.35)

        def first_or_zero_m(month_val: str, tag_val: str, col: str) -> float:
            ser = df_month.loc[(df_month["month"] == month_val) & (df_month["tag"] == tag_val), col]
            if ser.empty:
                return 0.0
            try:
                return float(ser.iloc[0])
            except Exception:
                return 0.0

        for i, t in enumerate(tags):
            offset = (i - (n - 1) / 2) * width
            y = [first_or_zero_m(m, t, "mean_ret") for m in months_x]
            plt.bar(x_idx + offset, y, width, label=t)

        plt.xticks(x_idx, months_x, rotation=45, ha="right")
        plt.legend()
    plt.title("Monthly mean return by tag")
    plt.xlabel("month")
    plt.ylabel("mean return")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "compare_monthly_mean.png"))
    print("[INFO] wrote compare_monthly_mean.png")

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
