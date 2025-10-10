# -*- coding: utf-8 -*-
import argparse, json, os, sys, math, warnings
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

"""
目的：
- trainset_{DatasetId}.csv を読み込み、BUY/SELLそれぞれのヒット確率を学習
- pカットオフをEV最大化で決める
- exports/ にモデル・閾値を保存
- （任意）config\stream_settings.json を更新

前提（緩やかに自動検出）：
- 目的変数（ラベル）候補：['label', 'label_hit', 'hit', 'y', 'target']
- 収益bp列候補：['ret_bp', 'pnl_bp', 'delta_bp', 'outcome_bp']
- グループ（CV分割）候補：['trade_date','date','session']
- 種別列：['side']（'BUY'/'SELL' が入っている想定）。無ければ --kind 指定でフィルタ。
- 特徴量：数値列から上記を除外して自動選別
"""


def pick_first(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None


def detect_columns(df):
    y_col = pick_first(df, ["label", "label_hit", "hit", "y", "target"])
    ret_col = pick_first(df, ["ret_bp", "pnl_bp", "delta_bp", "outcome_bp"])
    side_col = pick_first(df, ["side", "Side"])
    date_col = pick_first(df, ["trade_date", "date", "Date", "session"])
    return y_col, ret_col, side_col, date_col


def numeric_features(df, drop_cols):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num if c not in drop_cols]


def ev_from_mask(ret_series, mask):
    if mask.sum() == 0:
        return -1e9, 0.0
    sel = ret_series[mask]
    return sel.mean(), mask.mean()  # 平均bp, 採用率


def search_best_pcut(y_true, proba, ret_bp, grid=None):
    if grid is None:
        grid = np.linspace(0.40, 0.90, 21)  # 0.40〜0.90
    best = (-1e9, 0.0, 0.5)  # ev, fillrate, p*
    for pcut in grid:
        mask = proba >= pcut
        ev, fill = ev_from_mask(ret_bp, mask)
        if ev > best[0]:
            best = (ev, fill, float(pcut))
    return {
        "ev_bp": float(best[0]),
        "fill_rate": float(best[1]),
        "p_cut": float(best[2]),
    }


def suggest_thresholds(df_sel, feature_map):
    """
    高確率サンプル（p>=p*）から「それっぽい」しきい値を分位点で推定。
    feature_map: {'UPTICK':'uptick','SPREAD':'spread','SCORE':'score'}
    """
    out = {}
    for key, col in feature_map.items():
        if col in df_sel.columns:
            q = df_sel[col].quantile(0.30)  # 下30%を弾く程度の目安
            out[key] = float(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-DatasetId", required=True)
    ap.add_argument(
        "-CSV", help="trainset CSV 明示指定（省略時は trainset_{DatasetId}.csv を自動）"
    )
    ap.add_argument("-Kind", choices=["BUY", "SELL", "BOTH"], default="BOTH")
    ap.add_argument("-Model", choices=["xgb", "logit"], default="xgb")
    ap.add_argument("-CV", type=int, default=5)
    ap.add_argument("-OutDir", default="exports")
    ap.add_argument("-Config", default="config/stream_settings.json")
    ap.add_argument(
        "-EmitThresholds",
        action="store_true",
        help="確率フィルタに加えてしきい値提案も出力",
    )
    args = ap.parse_args()

    root = Path(".")
    out = Path(args.OutDir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.CSV) if args.CSV else Path(f"trainset_{args.DatasetId}.csv")
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    df0 = pd.read_csv(csv_path)
    y_col, ret_col, side_col, date_col = detect_columns(df0)
    if y_col is None:
        raise SystemExit("label列が見つかりません（例：label/label_hit/hit/y/target）")
    if ret_col is None:
        # ラベルから擬似bpを作る（平均利益・損失を推定）
        mu_hit = 120.0
        mu_loss = -20.0
        ret_col = "__ret_bp__"
        df0[ret_col] = np.where(df0[y_col].astype(int) == 1, mu_hit, mu_loss)

    if args.Kind != "BOTH":
        if side_col and side_col in df0.columns:
            df0 = df0[df0[side_col].str.upper() == args.Kind]
        else:
            # サイド列がない場合はそのまま全行を対象（片側CSV想定）
            pass

    drops = set([y_col, ret_col])
    if side_col:
        drops.add(side_col)
    if date_col:
        drops.add(date_col)
    X_cols = numeric_features(df0, drop_cols=drops)
    if not X_cols:
        raise SystemExit("数値の特徴量が見つかりません。")

    # モデル
    if args.Model == "xgb" and HAS_XGB:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=0,
        )
    else:
        # ロジスティック回帰（L2）
        base = LogisticRegression(max_iter=200, n_jobs=None, class_weight="balanced")
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)

    # CV分割
    if date_col and date_col in df0.columns:
        groups = df0[date_col]
    else:
        # 日単位グループが無い場合は等分割
        groups = pd.Series(np.arange(len(df0)) // max(1, len(df0) // args.CV))

    gkf = GroupKFold(n_splits=args.CV)
    rows = []
    oof_proba = np.zeros(len(df0), dtype=float)

    for fold, (tr, va) in enumerate(gkf.split(df0, groups=groups)):
        X_tr, y_tr = df0.iloc[tr][X_cols], df0.iloc[tr][y_col].astype(int)
        X_va, y_va = df0.iloc[va][X_cols], df0.iloc[va][y_col].astype(int)
        m = model
        m.fit(X_tr, y_tr)
        p = (
            m.predict_proba(X_va)[:, 1]
            if hasattr(m, "predict_proba")
            else m.decision_function(X_va)
        )
        oof_proba[va] = p
        auc = roc_auc_score(y_va, p)
        res = search_best_pcut(y_va.values, p, df0.iloc[va][ret_col].values)
        rows.append({"fold": fold, "auc": auc, **res})

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(
        out / f"grid_report_ml_{args.DatasetId}.csv", index=False, encoding="utf-8-sig"
    )

    # 全体最適 p*
    best_idx = cv_df["ev_bp"].idxmax()
    best_p = float(cv_df.loc[best_idx, "p_cut"])
    best_ev = float(cv_df.loc[best_idx, "ev_bp"])
    mean_auc = float(cv_df["auc"].mean())

    # 全データで再学習→保存
    model_final = model
    model_final.fit(df0[X_cols], df0[y_col].astype(int))
    # どちらのsideか
    side_tag = args.Kind if args.Kind in ["BUY", "SELL"] else "BOTH"
    dump(model_final, out / f"model_{side_tag}.joblib")

    # 高確率サンプルでしきい値提案（任意）
    thr_suggest = {}
    if args.EmitThresholds:
        df0 = df0.copy()
        proba_all = (
            model_final.predict_proba(df0[X_cols])[:, 1]
            if hasattr(model_final, "predict_proba")
            else model_final.decision_function(df0[X_cols])
        )
        df0["__proba__"] = proba_all
        sel = df0[df0["__proba__"] >= best_p]
        # よく使う列名をマップ（存在すれば拾う）
        feature_map = {
            "UPTICK_THR": next(
                (
                    c
                    for c in ["uptick", "uptick_thr", "tick_change", "dprice_tick"]
                    if c in df0.columns
                ),
                None,
            ),
            "SPREAD_MAX": next(
                (c for c in ["spread", "best_spread", "spr"] if c in df0.columns), None
            ),
            "SCORE_THR": next(
                (
                    c
                    for c in ["score", "rank_score", "signal_score"]
                    if c in df0.columns
                ),
                None,
            ),
        }
        feature_map = {k: v for k, v in feature_map.items() if v is not None}
        thr_suggest = suggest_thresholds(sel, feature_map)

    payload = {
        "dataset_id": args.DatasetId,
        "kind": args.Kind,
        "model": "xgboost" if (args.Model == "xgb" and HAS_XGB) else "logit",
        "cv_mean_auc": mean_auc,
        "best_prob_cut": best_p,
        "best_ev_bp": best_ev,
        "threshold_suggest": thr_suggest,
    }
    with open(
        out / f"thresholds_ml_{args.DatasetId}_{side_tag}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # stream_settings.json 更新（ML確率カットのみ既定で反映）
    try:
        cfg_path = Path(args.Config)
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            cfg = {}
        settings = cfg.setdefault("settings_naut", {})
        if args.Kind in ["BUY", "BOTH"]:
            settings["ML_PROB_CUT_BUY"] = best_p
        if args.Kind in ["SELL", "BOTH"]:
            settings["ML_PROB_CUT_SELL"] = best_p
        cfg_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] Updated {cfg_path} with ML_PROB_CUT_* = {best_p:.3f}")
    except Exception as e:
        print(f"[WARN] config update skipped: {e}", file=sys.stderr)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
