#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_grid_pipeline.py
  - dataset_id から DB を解決（dataset_registry）
  - labels / trainset を再作成
  - 分布とラベル偏りをダンプ（EV/precision 異常の原因を可視化）
  - grid_search_thresholds (BUY/SELL) を実行し結果要約
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def sh(cmd: list[str]) -> Tuple[int, str, str]:
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process.returncode, process.stdout, process.stderr


def resolve_db(dataset_id: str, registry_path: Path | None) -> Path:
    if registry_path and registry_path.exists():
        con = sqlite3.connect(str(registry_path))
        try:
            row = pd.read_sql(
                "SELECT db_path FROM dataset_registry WHERE dataset_id=? LIMIT 1",
                con,
                params=[dataset_id],
            )
            if not row.empty:
                return Path(row.iloc[0, 0])
        finally:
            con.close()
    guess = Path("db") / f"naut_market_{dataset_id[3:11]}_refeed.db"
    return guess


def stats_counts(db: Path, dataset_id: str) -> Dict[str, Dict[str, object]]:
    con = sqlite3.connect(str(db))
    out: Dict[str, Dict[str, object]] = {}
    try:
        table_names = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )["name"].tolist()

        def has_table(name: str) -> bool:
            return name in table_names

        def fetch_columns(name: str) -> list[str]:
            info = pd.read_sql(f"PRAGMA table_info({name});", con)
            return info["name"].tolist()

        def count(table: str, time_col: str | None, filter_dataset: bool = True) -> Dict[str, object]:
            if not has_table(table):
                return {"n": 0, "mn": None, "mx": None, "note": "no table"}

            if time_col:
                if filter_dataset and "dataset_id" in fetch_columns(table):
                    query = f"SELECT count(*) AS n, min({time_col}) AS mn, max({time_col}) AS mx FROM {table} WHERE dataset_id=?"
                    row = pd.read_sql(query, con, params=[dataset_id]).iloc[0]
                else:
                    query = f"SELECT count(*) AS n, min({time_col}) AS mn, max({time_col}) AS mx FROM {table}"
                    row = pd.read_sql(query, con).iloc[0]
            else:
                query = f"SELECT count(*) AS n FROM {table}"
                row = pd.read_sql(query, con).iloc[0]
            return row.to_dict()

        out["features_stream"] = count("features_stream", "t_exec", True)
        out["raw_push"] = (
            count("raw_push", "t_recv", False) if has_table("raw_push") else {"n": 0, "mn": None, "mx": None}
        )
        out["orderbook_snapshot"] = (
            count("orderbook_snapshot", "ts", False)
            if has_table("orderbook_snapshot")
            else {"n": 0, "mn": None, "mx": None}
        )
        return out
    finally:
        con.close()


def pretty(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def ensure_exports() -> Path:
    out_dir = Path("exports")
    out_dir.mkdir(exist_ok=True)
    return out_dir


def run(args: argparse.Namespace) -> int:
    dataset_id = args.DatasetId
    registry = Path(args.Registry) if args.Registry else Path("db") / "naut_market.db"
    registry_for_lookup = registry if registry.exists() else None
    db_path = Path(args.DB) if args.DB else resolve_db(dataset_id, registry_for_lookup)
    exports_dir = ensure_exports()

    print(f"[INFO] dataset_id = {dataset_id}")
    print(f"[INFO] registry   = {registry.resolve() if registry.exists() else '(not used)'}")
    print(f"[INFO] db         = {db_path.resolve()}")
    if not db_path.exists():
        print(f"[ERR] DB not found: {db_path}")
        return 2

    base_stats = stats_counts(db_path, dataset_id)
    print("[STATS: before]")
    print(pretty(base_stats))

    build_labels_cmd = [
        sys.executable,
        "-m",
        "scripts.build_labels_from_replay",
        "-DatasetId",
        dataset_id,
        "-Horizons",
        args.Horizons,
        "-Thresholds",
        args.Thresholds,
    ]
    print("\n[RUN]", " ".join(build_labels_cmd))
    rc, out, err = sh(build_labels_cmd)
    print(out)
    if rc != 0:
        if err:
            print(err, file=sys.stderr)
        return rc

    build_train_cmd = [sys.executable, "-m", "scripts.build_training_set", "-DatasetId", dataset_id]
    if args.ToleranceSec is not None:
        build_train_cmd += ["-ToleranceSec", str(args.ToleranceSec)]
    print("\n[RUN]", " ".join(build_train_cmd))
    rc, out, err = sh(build_train_cmd)
    print(out)
    if rc != 0:
        if err:
            print(err, file=sys.stderr)
        return rc

    train_csv = exports_dir / f"trainset_{dataset_id}.csv"
    if not train_csv.exists():
        print(f"[ERR] trainset CSV not found: {train_csv}")
        return 3

    df = pd.read_csv(train_csv)
    cols_exist = [col for col in ["label", "score", "f1", "f2", "f3", "spread_ticks"] if col in df.columns]
    if cols_exist:
        print("\n[TRAINSET describe]\n", df[cols_exist].describe().to_string())
    else:
        print("[WARN] No numerical columns available for describe().")

    if "label" in df.columns:
        vc = df["label"].value_counts(dropna=False)
        print("\n[label value_counts]\n", vc.to_string())
        pos = int(vc.get(1, 0))
        neg = int(vc.get(0, 0))
        total = len(df)
        ratio = (pos / total) if total else 0.0
        print(f"[label balance] pos={pos} neg={neg} total={total}  pos_ratio={ratio:.4f}")
    else:
        print("[WARN] 'label' column not found in trainset.")

    def run_grid(kind: str) -> int:
        module_name = "scripts.grid_search_thresholds" if kind == "BUY" else "scripts.grid_search_thresholds_sell"
        cmd = [sys.executable, "-m", module_name, "-DatasetId", dataset_id]
        if args.MinTrades:
            cmd += ["-MinTrades", str(args.MinTrades)]
        if args.EVFloor is not None:
            cmd += ["-EVFloor", str(args.EVFloor)]
        print("\n[RUN]", " ".join(cmd))
        rc, out, err = sh(cmd)
        print(out)
        if rc != 0 and err:
            print(err, file=sys.stderr)
        return rc

    if args.Kind in ("BUY", "BOTH"):
        run_grid("BUY")
    if args.Kind in ("SELL", "BOTH"):
        run_grid("SELL")

    def show_best(path: Path, tag: str) -> None:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = {
                key: data.get(key)
                for key in [
                    "dataset_id",
                    "precision",
                    "trades",
                    "ev",
                    "mean_hit_bp",
                    "mean_loss_bp",
                    "params",
                    "eligible",
                ]
            }
            print(f"\n[BEST {tag}]\n{pretty(summary)}")
        else:
            print(f"[WARN] not found: {path.name}")

    show_best(exports_dir / f"best_thresholds_{dataset_id}.json", "BUY")
    show_best(exports_dir / f"best_thresholds_sell_{dataset_id}.json", "SELL")

    after_stats = stats_counts(db_path, dataset_id)
    print("\n[STATS: after]")
    print(pretty(after_stats))
    print("\n[DONE] diagnose completed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose grid search pipeline for a dataset.")
    parser.add_argument("-DatasetId", required=True)
    parser.add_argument("-Registry", help="path to naut_market.db (for resolving db_path)")
    parser.add_argument("-DB", help="override refeed db path")
    parser.add_argument("-Horizons", default="60,120")
    parser.add_argument("-Thresholds", default="+8,-6")
    parser.add_argument("-ToleranceSec", type=float, default=3.0)
    parser.add_argument("-Kind", choices=["BUY", "SELL", "BOTH"], default="BOTH")
    parser.add_argument("-MinTrades", type=int, default=0)
    parser.add_argument("-EVFloor", type=float, default=None)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
