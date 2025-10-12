# -*- coding: utf-8 -*-
"""
visualize_naut_metrics.py
- 特徴量のヒートマップ/相関、グリッド候補のEV×precision×trades散布を一括生成
- 依存: pandas, numpy, matplotlib
- 使い方（PowerShell例はこの後に記載）

想定入力
- Trainset CSV（例: exports/trainset_REF20251010_OFF.csv）
  代表列: score, spread_ticks, f1, f2, f3, [label]
- Candidates CSV（例: exports/grid_candidates_REF20251010_OFF.csv）
  代表列: EV, precision, trades, eligible

オプションで列名は差し替え可能。
"""

import argparse
import os
import sys
import math
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        # fallback for cp932/shift-jis
        return pd.read_csv(path, encoding="cp932")


def fig_save(fig, out_png: str, pdf_pages: PdfPages):
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    pdf_pages.savefig(fig)
    plt.close(fig)


def plot_hist2d(
    df,
    xcol,
    ycol,
    outdir,
    title,
    bins=60,
    range_x=None,
    range_y=None,
    cmap="viridis",
    weights=None,
    pdf_pages=None,
):
    data = df[[xcol, ycol]].dropna()
    if data.empty:
        return None
    x = data[xcol].values
    y = data[ycol].values

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    h = ax.hist2d(
        x,
        y,
        bins=bins,
        range=[range_x, range_y] if (range_x and range_y) else None,
        cmap=cmap,
        weights=weights,
    )
    cb = fig.colorbar(h[3], ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    cb.set_label("count" if weights is None else "weighted count")

    out_png = os.path.join(outdir, f"{title.replace(' ', '_')}.png")
    fig_save(fig, out_png, pdf_pages)
    return out_png


def plot_corr(df, cols, outdir, title, pdf_pages=None):
    use_cols = [c for c in cols if c in df.columns]
    if len(use_cols) < 2:
        return None
    corr = df[use_cols].dropna().corr()
    if corr.empty:
        return None

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xticks(range(len(use_cols)))
    ax.set_yticks(range(len(use_cols)))
    ax.set_xticklabels(use_cols, rotation=45, ha="right")
    ax.set_yticklabels(use_cols)

    # 数値も重ね書き
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    out_png = os.path.join(outdir, f"{title.replace(' ', '_')}.png")
    fig_save(fig, out_png, pdf_pages)
    return out_png


def plot_scatter_sizecolor(
    df, xcol, ycol, sizecol, colorcol, outdir, title, pdf_pages=None
):
    data = df[[xcol, ycol, sizecol, colorcol]].dropna()
    if data.empty:
        return None

    x = data[xcol].values
    y = data[ycol].values
    s = data[sizecol].values
    c = data[colorcol].values

    # マーカーサイズ調整（極端な値の影響を抑制）
    s_scaled = 20 * (
        np.clip(s, np.percentile(s, 5), np.percentile(s, 95))
        / max(np.percentile(s, 95), 1.0)
    )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, s=s_scaled, c=c)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(colorcol)

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)

    out_png = os.path.join(outdir, f"{title.replace(' ', '_')}.png")
    fig_save(fig, out_png, pdf_pages)
    return out_png


def main():
    parser = argparse.ArgumentParser(description="DeepNaut 可視化一括スクリプト")
    parser.add_argument(
        "-Trainset",
        type=str,
        default=None,
        help="特徴量CSV（例: exports/trainset_*.csv）",
    )
    parser.add_argument(
        "-Candidates",
        type=str,
        default=None,
        help="グリッド候補CSV（例: exports/grid_candidates_*.csv）",
    )
    parser.add_argument(
        "-OutDir",
        type=str,
        default=None,
        help="出力先フォルダ（未指定なら exports\\viz_YYYYMMDD_HHMM）",
    )

    # 列名カスタム（trainset）
    parser.add_argument("-ScoreCol", type=str, default="score")
    parser.add_argument("-SpreadCol", type=str, default="spread_ticks")
    parser.add_argument("-FCols", type=str, default="f1,f2,f3")  # カンマ区切り
    parser.add_argument("-LabelCol", type=str, default=None)
    parser.add_argument(
        "-BuyValue",
        type=str,
        default=None,
        help="LabelColが指定された場合に抽出するBUY値（例: 1）",
    )

    # 列名カスタム（candidates）
    parser.add_argument("-EVCol", type=str, default="EV")
    parser.add_argument("-PrecCol", type=str, default="precision")
    parser.add_argument("-TradesCol", type=str, default="trades")
    parser.add_argument(
        "-EligibleCol", type=str, default="eligible"
    )  # bool/0-1/文字列OK

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = args.OutDir or os.path.join("exports", f"viz_{timestamp}")
    ensure_outdir(outdir)

    summary = {"outdir": outdir, "figures": []}
    pdf_path = os.path.join(outdir, "visual_summary.pdf")

    with PdfPages(pdf_path) as pdf:
        # ===== Trainset 可視化 =====
        if args.Trainset:
            train_df = safe_read_csv(args.Trainset)
            if (
                args.LabelCol
                and args.LabelCol in train_df.columns
                and args.BuyValue is not None
            ):
                train_df = train_df[
                    train_df[args.LabelCol].astype(str) == str(args.BuyValue)
                ]

            fcols = [c.strip() for c in args.FCols.split(",") if c.strip()]
            # 1) f2 vs f3 ヒートマップ
            if (
                "f2" in fcols
                and "f3" in fcols
                and "f2" in train_df.columns
                and "f3" in train_df.columns
            ):
                p = plot_hist2d(
                    train_df,
                    "f2",
                    "f3",
                    outdir,
                    "trainset_f2_vs_f3_heatmap",
                    bins=60,
                    pdf_pages=pdf,
                )
                if p:
                    summary["figures"].append(p)

            # 2) score vs spread_ticks ヒートマップ
            if args.ScoreCol in train_df.columns and args.SpreadCol in train_df.columns:
                p = plot_hist2d(
                    train_df,
                    args.ScoreCol,
                    args.SpreadCol,
                    outdir,
                    "trainset_score_vs_spread_ticks_heatmap",
                    bins=60,
                    pdf_pages=pdf,
                )
                if p:
                    summary["figures"].append(p)

            # 3) 相関ヒートマップ
            corr_cols = [args.ScoreCol, args.SpreadCol] + [c for c in fcols]
            p = plot_corr(
                train_df,
                corr_cols,
                outdir,
                "trainset_feature_correlation",
                pdf_pages=pdf,
            )
            if p:
                summary["figures"].append(p)

        # ===== Candidates 可視化 =====
        if args.Candidates:
            cand_df = safe_read_csv(args.Candidates)
            # eligible を 0/1 に正規化（列が無い／文字列でも吸収）
            if args.EligibleCol in cand_df.columns:
                elig = cand_df[args.EligibleCol]
                if elig.dtype == bool:
                    cand_df["_eligible_num"] = elig.astype(int)
                else:
                    cand_df["_eligible_num"] = (
                        elig.astype(str)
                        .str.lower()
                        .isin(["1", "true", "yes", "y", "✅"])
                        .astype(int)
                    )
            else:
                cand_df["_eligible_num"] = 1  # ない場合は全て適格扱い

            # 4) EV×precision 散布（サイズ=trades, 色=EV）
            if all(
                c in cand_df.columns for c in [args.EVCol, args.PrecCol, args.TradesCol]
            ):
                p = plot_scatter_sizecolor(
                    cand_df,
                    args.PrecCol,
                    args.EVCol,
                    args.TradesCol,
                    args.EVCol,
                    outdir,
                    "candidates_EV_vs_precision_scatter(size=trades,color=EV)",
                    pdf_pages=pdf,
                )
                if p:
                    summary["figures"].append(p)

            # 5) EV×precision の2Dヒート（重み=trades）
            if all(
                c in cand_df.columns for c in [args.EVCol, args.PrecCol, args.TradesCol]
            ):
                p = plot_hist2d(
                    cand_df,
                    args.PrecCol,
                    args.EVCol,
                    outdir,
                    "candidates_EV_vs_precision_hist2d_weighted_by_trades",
                    bins=60,
                    weights=cand_df[args.TradesCol].values,
                    pdf_pages=pdf,
                )
                if p:
                    summary["figures"].append(p)

            # 6) eligible 別の EV×precision 散布（色=eligible）
            if all(c in cand_df.columns for c in [args.EVCol, args.PrecCol]):
                p = plot_scatter_sizecolor(
                    cand_df,
                    args.PrecCol,
                    args.EVCol,
                    args.TradesCol if args.TradesCol in cand_df.columns else args.EVCol,
                    "_eligible_num",
                    outdir,
                    "candidates_EV_vs_precision_scatter(color=eligible)",
                    pdf_pages=pdf,
                )
                if p:
                    summary["figures"].append(p)

        # 空PDF回避：最低1枚は概要ページ
        if pdf.get_pagecount() == 0:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.02, 0.8, "visualize_naut_metrics", fontsize=14)
            ax.text(
                0.02,
                0.6,
                "※入力CSVが見つからないか、描画できる列がありませんでした。",
                fontsize=11,
            )
            fig_save(fig, os.path.join(outdir, "no_content.png"), pdf)

    # 実行サマリーJSON
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Figures saved to:", outdir)
    print(" PDF:", pdf_path)
    for p in summary["figures"]:
        print(" -", p)


if __name__ == "__main__":
    sys.exit(main())
