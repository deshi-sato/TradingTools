# scripts/score_tuner.py
# 前日スコア順位と翌日上昇率順位の順位相関を評価する最小実用版。
# ・DB から「銘柄数 = --count の最新日」と「その前日」を検出
# ・前日までの履歴でスコアを付与（score_table の関数が使えればそれを使用／無ければ簡易スコア）
# ・翌日の上昇率（前日終値→最新日終値）を計算
# ・両者の順位相関（Spearman / Kendall）を出力し、CSV を保存
#
# 使い方(例):
#   python scripts/score_tuner.py --db data/rss_data.db --count 332 --rank spearman --out results/score_tuner.csv
#   # テーブル・列名が異なる場合:
#   python scripts/score_tuner.py --db data/rss_data.db --count 332 --table quote_latest --date_col date --code_col code --price_col close --volume_col volume

from __future__ import annotations
import argparse
import os
import sys
import sqlite3
from typing import Optional, Dict, Tuple, List

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau


# --- score_table 連携（あれば使う / 無ければ簡易スコアで代替） -----------------
def _try_import_scorers():
    """
    score_table.py に長期・短期のスコア関数があれば読み込む。
    返り値: (long_fn, short_fn) いずれも None 可
      期待シグネチャ(柔軟にハンドリングする):
        fn(closes: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> (float or dict)
      - 数値を返す or dict の場合は dict.get("total", 0) を優先
    """
    long_fn = short_fn = None
    try:
        import score_table  # プロジェクト直下で実行すれば import 可

        # よくある関数名の候補
        for name in ["evaluate_stock_long", "score_long", "eval_long"]:
            if hasattr(score_table, name):
                long_fn = getattr(score_table, name)
                break
        for name in ["evaluate_stock_short", "score_short", "eval_short"]:
            if hasattr(score_table, name):
                short_fn = getattr(score_table, name)
                break
    except Exception:
        pass
    return long_fn, short_fn


def _call_score_fn(fn, closes, highs, lows, volumes) -> float:
    """関数呼び出しを頑健に。数値 or dict{'total': ...} を受け付ける。"""
    try:
        val = fn(closes, highs, lows, volumes)  # 最も一般的
    except TypeError:
        try:
            # 引数が少ない／多いケースにも一応対応
            val = fn(closes, volumes)  # 例: (closes, volumes)
        except Exception:
            val = None
    if isinstance(val, dict):
        return float(val.get("total", 0.0))
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    return 0.0


def _simple_score(closes, highs, lows, volumes) -> float:
    """
    フォールバックの簡易スコア:
      ・直近3日の終値モメンタム
      ・直近出来高の前年比率
      ・直近の終値位置（高値-安値レンジ内のどこか）
    ※ あくまで“仮のスコア”。プロジェクトの関数が使えればそちらが優先。
    """
    score = 0.0
    if len(closes) >= 4:
        # 3日モメンタム
        mom = closes[-1] / closes[-4] - 1.0
        score += 2.0 * mom
    if len(volumes) >= 2 and volumes[-2] > 0:
        vratio = (volumes[-1] - volumes[-2]) / volumes[-2]
        score += np.clip(vratio, -0.2, 0.2)  # ほどほどにクリップ
    if highs and lows and highs[-1] > lows[-1]:
        pos = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1])
        score += (pos - 0.5) * 0.5
    return float(score)


# --- DB / カラム検出 ----------------------------------------------------------
def _lower_map(columns: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}


def _get_tables(conn: sqlite3.Connection) -> List[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    return [r[0] for r in conn.execute(q).fetchall()]


def _get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    q = f"PRAGMA table_info('{table}')"
    return [r[1] for r in conn.execute(q).fetchall()]


def _pick_first(col_map: Dict[str, str], cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in col_map:
            return col_map[c]
    return None


def _find_core_table(
    conn: sqlite3.Connection,
) -> Tuple[str, str, str, str, Optional[str]]:
    """
    date, code, price(close), volume を持つ可能性が高いテーブルと列を選ぶ。
    返り値: (table, date_col, code_col, price_col, volume_col)
    volume_col は無くてもOK(None)
    """
    best = None
    tables = _get_tables(conn)
    for t in tables:
        cols = _get_table_columns(conn, t)
        lm = _lower_map(cols)
        date_col = _pick_first(lm, ["date", "trade_date", "dt"])
        code_col = _pick_first(lm, ["code", "ticker", "symbol", "stock_code", "secid"])
        price_col = _pick_first(
            lm, ["close", "adj_close", "adjusted_close", "last", "price"]
        )
        volume_col = _pick_first(lm, ["volume", "vol", "turnover"])
        # スコア: date+code+price を最優先, volume は加点
        score = 0
        if date_col and code_col:
            score += 10
        if price_col:
            score += 10
        if volume_col:
            score += 1
        if score >= 20:
            best = (t, date_col, code_col, price_col, volume_col)
            break
    if not best:
        raise RuntimeError(
            "date/code/close を含むテーブルが見つかりません。--table/--*_col で指定してください。"
        )
    return best


# --- メインロジック -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="前日スコア順位と翌日上昇率順位の順位相関を評価"
    )
    ap.add_argument("--db", type=str, required=True, help="SQLite DB パス")
    ap.add_argument(
        "--count", type=int, default=332, help="当日の銘柄数（完全一致の日付を採用）"
    )
    ap.add_argument(
        "--rank",
        type=str,
        default="spearman",
        choices=["spearman", "kendall"],
        help="相関指標",
    )
    ap.add_argument(
        "--out", type=str, default="results/score_tuner.csv", help="保存先 CSV"
    )
    # テーブル・列を明示指定したい場合
    ap.add_argument("--table", type=str, default=None)
    ap.add_argument("--date_col", type=str, default=None)
    ap.add_argument("--code_col", type=str, default=None)
    ap.add_argument("--price_col", type=str, default=None)
    ap.add_argument("--volume_col", type=str, default=None)
    # 履歴に使う最大日数（計算効率のため制限）
    ap.add_argument(
        "--lookback", type=int, default=120, help="前日スコア計算に使う最大過去日数"
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    conn = sqlite3.connect(args.db)
    try:
        # テーブル・列名の決定
        if args.table and args.date_col and args.code_col and args.price_col:
            table = args.table
            date_col = args.date_col
            code_col = args.code_col
            price_col = args.price_col
            volume_col = args.volume_col
        else:
            table, date_col, code_col, price_col, volume_col = _find_core_table(conn)

        # 最新日(= count 件ある最新) と その前日 を特定
        q = f"""
            SELECT {date_col} AS d, COUNT(*) AS n
            FROM {table}
            GROUP BY {date_col}
            HAVING n = ?
            ORDER BY d DESC
            LIMIT 2
        """
        dates = conn.execute(q, (args.count,)).fetchall()
        if len(dates) < 2:
            raise RuntimeError(
                f"銘柄数 {args.count} 件の日付が連続する 2 日分見つかりません。--count を調整するか列指定を見直してください。"
            )
        latest_date, prev_date = dates[0][0], dates[1][0]

        # 前日までの履歴をロード（lookback で制限）
        q_hist = f"""
            SELECT {date_col} AS d, {code_col} AS code, {price_col} AS close
                   {(',' if volume_col else '') + (volume_col + ' AS volume' if volume_col else '')}
            FROM {table}
            WHERE d <= ?
              AND d >= (SELECT d FROM (
                          SELECT DISTINCT {date_col} AS d
                          FROM {table}
                          WHERE {date_col} <= ?
                          ORDER BY d DESC
                          LIMIT ?
                        ) ORDER BY d ASC LIMIT 1)
        """
        df_hist = pd.read_sql_query(
            q_hist, conn, params=(prev_date, prev_date, args.lookback)
        )
        if df_hist.empty:
            raise RuntimeError("履歴データが取得できませんでした。")

        # 最新日と前日の終値も取得（上昇率用）
        q_day = f"""
            SELECT {code_col} AS code, {price_col} AS close
            FROM {table}
            WHERE {date_col} = ?
        """
        df_prev = pd.read_sql_query(q_day, conn, params=(prev_date,))
        df_latest = pd.read_sql_query(q_day, conn, params=(latest_date,))

        # スコア関数の取得（無ければ簡易スコア）
        long_fn, short_fn = _try_import_scorers()
        use_simple = long_fn is None and short_fn is None

        # 前日スコア算出（コード毎に履歴を取り出して評価）
        scores = []
        g = df_hist.groupby("code", sort=False)
        for code, gdf in g:
            gdf = gdf.sort_values("d")
            closes = gdf["close"].tolist()
            highs = lows = (
                []
            )  # DB からは取っていないので空（score_fn が無くても安全に動くように）
            volumes = gdf["volume"].tolist() if "volume" in gdf.columns else []
            if len(closes) < 5:
                continue

            if not use_simple and long_fn is not None:
                s_long = _call_score_fn(long_fn, closes, highs, lows, volumes)
            else:
                s_long = _simple_score(closes, highs, lows, volumes)

            if not use_simple and short_fn is not None:
                s_short = _call_score_fn(short_fn, closes, highs, lows, volumes)
            else:
                s_short = -_simple_score(closes, highs, lows, volumes)  # 簡易の逆張り

            scores.append(
                {"code": code, "prev_score_long": s_long, "prev_score_short": s_short}
            )

        df_score = pd.DataFrame(scores)
        if df_score.empty:
            raise RuntimeError("スコアが計算できませんでした（履歴が足りない等）。")

        # 翌日の上昇率を計算
        df_merge = (
            df_prev.rename(columns={"close": "prev_close"})
            .merge(
                df_latest.rename(columns={"close": "latest_close"}),
                on="code",
                how="inner",
            )
            .merge(df_score, on="code", how="inner")
        )
        df_merge["next_return"] = (
            df_merge["latest_close"] / df_merge["prev_close"] - 1.0
        )

        # 順位相関
        # 翌日の上昇率の降順順位
        rank_return = df_merge["next_return"].rank(ascending=False, method="average")
        # 前日スコアの降順順位（ロング/ショート両方評価）
        rank_long = df_merge["prev_score_long"].rank(ascending=False, method="average")
        rank_short = df_merge["prev_score_short"].rank(
            ascending=False, method="average"
        )

        if args.rank == "spearman":
            corr_long = spearmanr(rank_return, rank_long, nan_policy="omit").correlation
            corr_short = spearmanr(
                rank_return, rank_short, nan_policy="omit"
            ).correlation
        else:
            corr_long = kendalltau(rank_return, rank_long).correlation
            corr_short = kendalltau(rank_return, rank_short).correlation

        # 出力
        print(
            f"Table: {args.table or '(auto)'}  Date(prev, latest)=({prev_date}, {latest_date})  Samples: {len(df_merge)}"
        )
        print(f"{args.rank.title()}  long vs next:  {corr_long:.6f}")
        print(f"{args.rank.title()}  short vs next: {corr_short:.6f}")

        # CSV 保存（翌日上昇率で降順）
        out_df = df_merge[
            ["code", "prev_score_long", "prev_score_short", "next_return"]
        ].copy()
        out_df = out_df.sort_values("next_return", ascending=False)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Saved: {args.out}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
