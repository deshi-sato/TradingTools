# -*- coding: utf-8 -*-
"""
score_tuner.py  (daily-ready, summary + per-code detail exporter)

日足OHLCVを使って Prev(日)の3因子スコアを作成し、Next(日)のリターンと整合度を評価。
出力:
- サマリー: --out で指定 (例: data/score_daily.csv)
- 明細    : --out のベース名 + ".codes.csv"  (例: data/score_daily.codes.csv)
           列 = [date, code, score, next_return]
"""

from __future__ import annotations
import argparse, os, sqlite3
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


# ========= 1) データ取得 =========
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
    """SQLiteから必要列を取得（dt, code, close, volume, high）。日足テーブルでもOK。"""
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
    return df.sort_values(["code", "dt"]).reset_index(drop=True)


def minutes_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """標準列(dt, code, close, volume, high) → 日足(date, code, close, volume, high)"""
    d = df[["dt", "code", "close", "volume", "high"]].copy()
    d["date"] = d["dt"].dt.date
    daily = (
        d.groupby(["date", "code"], as_index=False)
        .agg(close=("close", "last"), volume=("volume", "sum"), high=("high", "max"))
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )
    return daily


def build_prev_next_pairs(daily: pd.DataFrame) -> pd.DataFrame:
    """各銘柄ごとに Prev→Next ペアを作る（Prevでスコア、Nextでリターン）。"""
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)

    def _per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["date_next"] = g["date"].shift(-1)
        g["close_next"] = g["close"].shift(-1)
        g["high_prevday"] = g["high"].shift(1)
        g["next_return"] = (g["close_next"] - g["close"]) / g["close"]
        return g

    paired = (
        daily.groupby("code", group_keys=False)
        .apply(_per_code)
        .dropna(subset=["date_next"])
        .reset_index(drop=True)
    )
    return paired


# ========= 2) 因子スコア（Prev側で計算） =========
def compute_factor_scores(paired: pd.DataFrame) -> pd.DataFrame:
    """
    3因子:
      score_vol = max(vol/MA5(vol)-1, 0)
      score_ma  = 1{ close > MA5(close) }
      score_brk = 1{ close >= 前日高値 }
    """
    df = paired.copy()

    def _per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ma5"] = g["close"].rolling(5, min_periods=3).mean()
        g["vol_mean5"] = g["volume"].rolling(5, min_periods=3).mean()
        ratio = g["volume"] / g["vol_mean5"]
        g["score_vol"] = np.clip(ratio - 1.0, 0.0, None).astype(float)
        g["score_ma"] = (g["close"] > g["ma5"]).astype(float)
        g["score_brk"] = (g["close"] >= g["high_prevday"]).astype(float)
        return g

    df = df.groupby("code", group_keys=False).apply(_per_code)
    for c in ["score_vol", "score_ma", "score_brk"]:
        df[c] = df[c].fillna(0.0)
    return df


# ========= 3) 評価ロジック =========
def evaluate_by_date(
    df_scores: pd.DataFrame,
    w: Tuple[float, float, float],
    dates: List[str],
    rank_method: str = "spearman",
    topn: int | None = None,
) -> Tuple[float, float, int]:
    corr_list, top_list = [], []
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


# ========= 4) 明細エクスポート（共通） =========
def export_codes_detail(
    scored: pd.DataFrame,
    w: Tuple[float, float, float],
    use_dates: List[str],
    out_base: str,
) -> str:
    """評価に使った全日付について [date,code,score,next_return] を保存。"""
    import os

    detail = scored[scored["date"].isin(use_dates)].copy()
    detail["score"] = (
        w[0] * detail["score_vol"]
        + w[1] * detail["score_ma"]
        + w[2] * detail["score_brk"]
    )
    detail = detail[["date", "code", "score", "next_return"]].copy()
    detail["date"] = detail["date"].astype(str)
    root, _ = os.path.splitext(out_base if out_base else "data/score_daily.csv")
    out_codes = root + ".codes.csv"
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    detail.to_csv(out_codes, index=False, encoding="utf-8")
    print(f"Saved per-code detail: {out_codes} rows: {len(detail)}")
    return out_codes


# ========= 5) メイン =========
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

    # 1) 取得→日足化
    minutes = fetch_minutes_as_df(
        args.db,
        args.table,
        args.date_col,
        args.code_col,
        args.price_col,
        args.volume_col,
        args.start,
        args.end,
        pad_days=5,
    )
    daily = minutes_to_daily(minutes)

    # 2) Prev→Next & 因子
    paired = build_prev_next_pairs(daily)
    scored = compute_factor_scores(paired)

    # 3) 期間フィルタ（Prev日ベース）
    start_date = pd.to_datetime(args.start).date()
    end_date = pd.to_datetime(args.end).date()
    scored = scored.loc[
        (scored["date"] >= start_date) & (scored["date"] <= end_date)
    ].copy()
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
        # === 通常評価 ===
        w = (1.0, 1.0, 1.0)
        avg_corr, avg_top, used = evaluate_by_date(
            scored, w, use_dates, args.rank, args.topn
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
        # ★ 明細の出力
        export_codes_detail(scored, w, use_dates, args.out)
        return

    # === 重み最適化 ===
    grid = np.arange(0.0, 1.0 + 1e-9, args.step)
    cand = [
        (float(a), float(b), float(c))
        for a in grid
        for b in grid
        for c in grid
        if not (a == 0 and b == 0 and c == 0)
    ]
    rows: List[Dict] = []
    for w in cand:
        avg_corr, avg_top, used = evaluate_by_date(
            scored, w, use_dates, args.rank, args.topn
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
    # ★ 最良重みで明細も出力
    w_best = (float(best.w_vol), float(best.w_ma), float(best.w_brk))
    export_codes_detail(scored, w_best, use_dates, args.out)


if __name__ == "__main__":
    main()
