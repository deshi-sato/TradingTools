# -*- coding: utf-8 -*-
# 明細CSV (data/score_daily.codes.csv) を優先して解析。
# フォールバック: data/score_daily.csv（旧フォーマット）
# 出力: data/analysis/{summary.txt, ic_by_day.csv, topn_by_day.csv, ic_hist.png, top10_cum.png}

import os, re, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = r"data/analysis"
os.makedirs(OUTDIR, exist_ok=True)

# 1) 入力ファイルの自動選択
CODES = r"data/score_daily.codes.csv"
BASE  = r"data/score_daily.csv"
if os.path.exists(CODES):
    INP = CODES
    mode = "codes"
else:
    INP = BASE
    mode = "base"

if not os.path.exists(INP):
    print(f"[ERROR] not found: {INP}")
    sys.exit(1)

df = pd.read_csv(INP)
cols_lower = [c.lower() for c in df.columns]
orig_cols  = list(df.columns)

def pick(*cands):
    for name in cands:
        if name in cols_lower:
            return orig_cols[cols_lower.index(name)]
    return None

if mode == "codes":
    # 期待列: date, code, score, next_return
    date_col  = pick("date")
    id_col    = pick("code","ticker","symbol")
    score_col = pick("score","prev_score_short","prev_score_long","signal","pred")
    ret_col   = pick("next_return","ret_next","return_next","ret","target")
else:
    # 旧: base（列に score/ret が無いことが多い）
    date_col  = pick("date")
    id_col    = pick("code","ticker","symbol")
    score_col = pick("score","prev_score_short","prev_score_long","signal","pred")
    ret_col   = pick("next_return","ret_next","return_next","ret","target")

# 型整形
if date_col: df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
if score_col: df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
if ret_col: df[ret_col]     = pd.to_numeric(df[ret_col], errors="coerce")

# 2) codes モード（推奨）: 日別IC/TopN
if mode == "codes" and all([date_col, id_col, score_col, ret_col]):
    def spearman_ic(g): return g[score_col].corr(g[ret_col], method="spearman")
    ic = df.groupby(date_col).apply(spearman_ic).rename("IC").dropna().reset_index()
    ic.to_csv(os.path.join(OUTDIR,"ic_by_day.csv"), index=False)

    topn_df = []
    for n in (5,10,20):
        s = df.groupby(date_col).apply(lambda g: g.sort_values(score_col, ascending=False).head(n)[ret_col].mean())
        s = s.rename(f"top{n}_ret")
        topn_df.append(s)
    topn_df = pd.concat(topn_df, axis=1).dropna().reset_index()
    topn_df.to_csv(os.path.join(OUTDIR,"topn_by_day.csv"), index=False)

    # サマリ
    summary = []
    summary.append(f"MODE=A(codes)  rows={len(df)}  days={len(ic)}  from {ic[date_col].min()} to {ic[date_col].max()}")
    summary.append(f"IC mean={ic['IC'].mean():.6f}  std={ic['IC'].std():.6f}")
    for n in (5,10,20):
        summary.append(f"Top{n} mean next-ret = {topn_df[f'top{n}_ret'].mean():.6f}")
    open(os.path.join(OUTDIR,"summary.txt"),"w",encoding="utf-8").write("\n".join(summary))
    print("\n".join(summary))

    # 図（色指定なし）
    plt.figure()
    plt.hist(ic["IC"].dropna(), bins=30)
    plt.title("IC (Spearman) Histogram"); plt.xlabel("IC"); plt.ylabel("Freq")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"ic_hist.png"))

    plt.figure()
    (topn_df.set_index(date_col)["top10_ret"].fillna(0).cumsum()).plot()
    plt.title("Top10 Next-Day Return (Cumulative)"); plt.xlabel("Date"); plt.ylabel("Cum. Return")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"top10_cum.png"))

# 3) base モード: スナップショットのみ（旧互換）
else:
    # 必須が揃っていない場合はスナップショットにフォールバック
    print(f"[WARN] fall back to snapshot mode. cols: date={date_col}, id={id_col}, score={score_col}, ret={ret_col}")
    summary = []
    summary.append(f"MODE=B(snapshot)  rows={len(df)}")
    if score_col and ret_col:
        ic = float(df[score_col].corr(df[ret_col], method="spearman"))
        summary.append(f"IC (Spearman) = {ic:.6f}")
        top = {}
        for n in (5,10,20):
            top[f"top{n}_ret"] = df.sort_values(score_col, ascending=False).head(n)[ret_col].mean()
            summary.append(f"Top{n} mean next-ret = {top[f'top{n}_ret']:.6f}")
        pd.DataFrame([top]).to_csv(os.path.join(OUTDIR,"topn_snapshot.csv"), index=False)
    else:
        summary.append("score/next_return が不足のため snapshot 計算不可")
    open(os.path.join(OUTDIR,"summary.txt"),"w",encoding="utf-8").write("\n".join(summary))
    print("\n".join(summary))
