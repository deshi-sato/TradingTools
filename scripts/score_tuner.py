# -*- coding: utf-8 -*-
"""
score_tuner.py

1分足のSQLiteテーブルから日足OHLCVを集計し、
「出来高急増」「5日線上抜け」「前日高値ブレイク」の3因子を
前日(Prev)に算出、翌日(Next)のリターンと整合度を評価します。

通常評価:
  python scripts\score_tuner.py --db data\rss_data.db --table minute_data --date_col datetime --code_col ticker --price_col close --volume_col volume --start 2025-08-14 --end 2025-08-15 --out results\score_tuner_summary.csv

重み最適化:
  python scripts\score_tuner.py --db data\rss_data.db --table minute_data --date_col datetime --code_col ticker --price_col close --volume_col volume --start 2025-08-14 --end 2025-08-15 --optimize --step 0.25 --topn 10 --out results\weights_grid.csv
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


# ========= 1分足取り出し → 日足集計 =========


def fetch_minutes_as_df(
    db_path: str,
    table: str,
    date_col: str,
    code_col: str,
    price_col: str,
    volume_col: str,
    start: str,
    end: str,
    pad_days: int = 5,
) -> pd.DataFrame:
    """
    SQLiteの1分足から必要列を取得し、標準列名に正規化する。
    取り出す列: dt, code, close, volume, high
    """
    conn = sqlite3.connect(db_path)
    try:
        q = f"""
            SELECT
              {date_col}  AS dt,
              {code_col}  AS code,
              {price_col} AS close,
              {volume_col} AS volume,
              high        AS high
            FROM {table}
            WHERE date({date_col}) BETWEEN date(?,'-{pad_days} day') AND date(?)
        """
        df = pd.read_sql_query(q, conn, params=[start, end])
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError("指定範囲にレコードがありません (--start/--end を確認)。")

    df["dt"] = pd.to_datetime(df["dt"])
    df["date"] = df["dt"].dt.date.astype(str)
    df = df.sort_values(["code", "dt"]).reset_index(drop=True)
    return df


def minutes_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    標準化された1分足(df: dt, code, close, volume, high)から日足に集計。
    """
    d = df[["dt", "code", "close", "volume", "high"]].copy()
    d["date"] = d["dt"].dt.date

    daily = (
        d.groupby(["date", "code"], as_index=False)
        .agg(
            close=("close", "last"),
            volume=("volume", "sum"),
            high=("high", "max"),
        )
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )
    return daily


def build_prev_next_pairs(daily: pd.DataFrame) -> pd.DataFrame:
    """
    各銘柄ごとに(Prev, Next)ペアを作る。Prevはスコア計算、Nextは翌日リターン。
    """
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)

    def _per_code(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date_next"] = df["date"].shift(-1)
        df["close_next"] = df["close"].shift(-1)
        df["high_prevday"] = df["high"].shift(1)  # 前日高値（Prev基準）
        df["next_return"] = (df["close_next"] - df["close"]) / df["close"]
        return df

    paired = (
        daily.groupby("code", group_keys=False)
        .apply(_per_code)
        .dropna(subset=["date_next"])
        .reset_index(drop=True)
    )
    return paired


# ========= 3因子スコア =========


def compute_factor_scores(paired: pd.DataFrame) -> pd.DataFrame:
    """
    3因子をPrev側で算出:
      - score_vol: 出来高急増 (vol/5日平均 - 1).clip(min=0)
      - score_ma : 5日線上抜け (close > MA5)
      - score_brk: 前日高値ブレイク (close >= high_prevday)
    """
    df = paired.copy()

    def _per_code(df1: pd.DataFrame) -> pd.DataFrame:
        df1 = df1.copy()
        df1["ma5"] = df1["close"].rolling(5, min_periods=3).mean()
        df1["vol_mean5"] = df1["volume"].rolling(5, min_periods=3).mean()

        ratio = df1["volume"] / df1["vol_mean5"]
        df1["score_vol"] = np.clip(ratio - 1.0, 0.0, None).astype(float)
        df1["score_ma"] = (df1["close"] > df1["ma5"]).astype(float)
        df1["score_brk"] = (df1["close"] >= df1["high_prevday"]).astype(float)
        return df1

    df = df.groupby("code", group_keys=False).apply(_per_code)

    for c in ["score_vol", "score_ma", "score_brk"]:
        df[c] = df[c].fillna(0.0)

    return df


# ========= 評価ロジック =========


def evaluate_by_date(
    df_scores: pd.DataFrame,
    w: Tuple[float, float, float],
    dates: List[str],
    rank_method: str = "spearman",
    topn: int | None = None,
) -> Tuple[float, float, int]:
    """
    日ごとにスコア= w·f を作って順位相関/TopN平均を計算し、平均を返す。
    """
    corr_list: List[float] = []
    top_list: List[float] = []

    for d in dates:
        day = df_scores[df_scores["date"] == d]
        if day.empty:
            continue

        score = (
            w[0] * day["score_vol"] + w[1] * day["score_ma"] + w[2] * day["score_brk"]
        )

        try:
            if rank_method == "spearman":
                corr, _ = spearmanr(score, day["next_return"])
            else:
                corr, _ = kendalltau(score, day["next_return"])
        except Exception:
            corr = np.nan

        if not np.isnan(corr):
            corr_list.append(float(corr))

        if topn and topn > 0:
            top_df = (
                day.assign(score=score).sort_values("score", ascending=False).head(topn)
            )
            if not top_df.empty:
                top_list.append(float(np.mean(top_df["next_return"])))

    avg_corr = float(np.mean(corr_list)) if corr_list else float("nan")
    avg_top = float(np.mean(top_list)) if top_list else float("nan")
    return avg_corr, avg_top, len(corr_list)


# ========= メイン =========


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="minute_data")
    ap.add_argument("--date_col", default="datetime")
    ap.add_argument("--code_col", default="ticker")
    ap.add_argument("--price_col", default="close")
    ap.add_argument("--volume_col", default="volume")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (Prevの初日)")
    ap.add_argument(
        "--end", required=True, help="YYYY-MM-DD (Next最終日を含むペアのPrev最大日)"
    )
    ap.add_argument("--rank", choices=["spearman", "kendall"], default="spearman")
    ap.add_argument("--out", default="results\\score_tuner_summary.csv")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    # 1) 1分足→日足
    minutes = fetch_minutes_as_df(
        db_path=args.db,
        table=args.table,
        date_col=args.date_col,
        code_col=args.code_col,
        price_col=args.price_col,
        volume_col=args.volume_col,
        start=args.start,
        end=args.end,
        pad_days=5,
    )
    daily = minutes_to_daily(minutes)

    # 2) Prev→Nextペア & スコア
    paired = build_prev_next_pairs(daily)
    scored = compute_factor_scores(paired)

    # 3) 評価対象(Prev日)を期間でフィルタ
    start_date = pd.to_datetime(args.start).date()
    end_date = pd.to_datetime(args.end).date()

    mask = (scored["date"] >= start_date) & (scored["date"] <= end_date)
    scored = scored.loc[mask].copy()
    if scored.empty:
        raise RuntimeError(
            "評価できる日ペアがありませんでした (--start/--end を見直してください)。"
        )

    use_dates = sorted(scored["date"].unique().tolist())
    print(
        f"Table: {args.table}  Date(prev..end)=({use_dates[0]}, {use_dates[-1]})  Days: {len(use_dates)}"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not args.optimize:
        w = (1.0, 1.0, 1.0)
        avg_corr, avg_top, used = evaluate_by_date(
            scored, w=w, dates=use_dates, rank_method=args.rank, topn=args.topn
        )
        pd.DataFrame(
            [
                {
                    "w_vol": w[0],
                    "w_ma": w[1],
                    "w_brk": w[2],
                    "days_used": used,
                    "avg_corr": avg_corr,
                    "avg_topN_return": avg_top,
                    "rank_method": args.rank,
                }
            ]
        ).to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"{args.rank.title()}  avg corr: {avg_corr:.6f}")
        if args.topn > 0:
            print(f"Top{args.topn}  avg next_return: {avg_top:.6f}")
        print(f"Saved: {args.out}")
        return

    # ---- 重み最適化 ----
    grid_vals = np.arange(0.0, 1.0 + 1e-9, args.step)
    cand: List[Tuple[float, float, float]] = [
        (float(a), float(b), float(c))
        for a in grid_vals
        for b in grid_vals
        for c in grid_vals
        if not (a == 0 and b == 0 and c == 0)
    ]

    rows: List[Dict] = []
    for w in cand:
        avg_corr, avg_top, used = evaluate_by_date(
            scored, w=w, dates=use_dates, rank_method=args.rank, topn=args.topn
        )
        rows.append(
            {
                "w_vol": w[0],
                "w_ma": w[1],
                "w_brk": w[2],
                "days_used": used,
                "avg_corr": avg_corr,
                "avg_topN_return": avg_top,
            }
        )

    df_res = pd.DataFrame(rows).sort_values(
        ["avg_corr", "avg_topN_return"], ascending=[False, False]
    )
    df_res.to_csv(args.out, index=False, encoding="utf-8-sig")

    best = df_res.iloc[0]
    print("=== Best Weights ===")
    print(f"w_vol={best.w_vol:.2f}, w_ma={best.w_ma:.2f}, w_brk={best.w_brk:.2f}")
    print(
        f"avg_corr={best.avg_corr:.6f}, avg_topN_return={best.avg_topN_return:.6f}, days_used={int(best.days_used)}"
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
