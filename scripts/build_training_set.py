#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_training_set.py

Exports a vertical training CSV by joining labels_outcome with features_stream.
Uses dataset_registry to locate the correct dated refeed DB.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

JST_DEFAULT_COLS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "score",
    "spread_ticks",
    "bid1",
    "ask1",
    "bidqty1",
    "askqty1",
]


def parse_cols(arg: Optional[str]) -> List[str]:
    if not arg:
        return list(JST_DEFAULT_COLS)
    cols = [piece.strip() for piece in arg.split(",") if piece.strip()]
    if not cols:
        return list(JST_DEFAULT_COLS)
    return cols


def resolve_db_path(dataset_id: str) -> Path:
    db_dir = Path("db")
    candidates = sorted(db_dir.glob("naut_market_*_refeed.db"))
    if not candidates:
        raise SystemExit("ERROR: no refeed DBs under db/")
    for path in candidates:
        try:
            conn = sqlite3.connect(str(path))
            cur = conn.execute(
                "SELECT COALESCE(db_path, source_db_path) FROM dataset_registry WHERE dataset_id=?",
                (dataset_id,),
            )
            row = cur.fetchone()
            conn.close()
        except sqlite3.DatabaseError:
            continue
        if not row:
            continue
        source = row[0]
        final_path = Path(source) if source else path
        if not final_path.exists():
            final_path = path
        return final_path.resolve()
    raise SystemExit(f"ERROR: dataset_id {dataset_id} not found in registry.")


def ensure_indexes(conn: sqlite3.Connection) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_labels_dataset_symbol_ts ON labels_outcome(dataset_id, symbol, ts)",
        "CREATE INDEX IF NOT EXISTS idx_features_symbol_ts ON features_stream(symbol, t_exec)",
    ]
    for stmt in statements:
        conn.execute(stmt)
    conn.commit()


def load_tables_for_asof(
    conn: sqlite3.Connection,
    dataset_id: str,
    extra_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labels (left) and features (right) as DataFrame for asof-merge."""
    lab = pd.read_sql(
        """
        SELECT symbol, ts, horizon_sec, ret_bp, label
          FROM labels_outcome
         WHERE dataset_id=?
         ORDER BY symbol, ts
        """,
        conn,
        params=(dataset_id,),
    )
    feat_cols = ["symbol", "t_exec"] + list(extra_cols)
    feat = pd.read_sql(
        f"""
        SELECT {", ".join(feat_cols)}
          FROM features_stream
         ORDER BY symbol, t_exec
        """,
        conn,
    )
    # 型そろえ（秒単位を想定。ms で来たら秒に寄せる）
    lab["ts"] = pd.to_numeric(lab["ts"], errors="coerce").astype("float64")
    feat["t_exec"] = pd.to_numeric(feat["t_exec"], errors="coerce").astype("float64")
    # ms 判定（だいたい 1e11 以上は ms）→ 秒に変換
    if not lab["ts"].dropna().empty and lab["ts"].dropna().iloc[0] > 1e11:
        lab["ts"] = lab["ts"] / 1000.0
    if not feat["t_exec"].dropna().empty and feat["t_exec"].dropna().iloc[0] > 1e11:
        feat["t_exec"] = feat["t_exec"] / 1000.0
    return lab, feat


def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(
    out_path: Path,
    df: pd.DataFrame,
) -> int:
    ensure_output_path(out_path)
    df.to_csv(out_path, index=False, encoding="utf-8-sig", lineterminator="\r\n")
    return len(df)


def run(args: argparse.Namespace) -> None:
    dataset_id = args.DatasetId
    extra_cols = parse_cols(args.Cols)
    out_path = Path(args.Out) if args.Out else Path(f"exports/trainset_{dataset_id}.csv")

    db_path = resolve_db_path(dataset_id)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    has_labels = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='labels_outcome'"
    ).fetchone()
    if not has_labels:
        conn.close()
        raise SystemExit("ERROR: labels_outcome missing → run build_labels_from_replay first.")
    ensure_indexes(conn)

    # --- 近傍JOIN（asof）に変更 ---
    lab, feat = load_tables_for_asof(conn, dataset_id, extra_cols)
    # 許容秒は引数で可変（デフォルト 3.0）
    tol = float(args.ToleranceSec)
    dfs: list[pd.DataFrame] = []
    for sym, g_lab in lab.groupby("symbol", sort=False):
        g_feat = feat[feat["symbol"] == sym]
        if g_feat.empty:
            continue
        m = pd.merge_asof(
            g_lab.sort_values("ts"),
            g_feat.sort_values("t_exec"),
            left_on="ts",
            right_on="t_exec",
            direction="nearest",
            tolerance=tol,
        )
        dfs.append(m)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    # マッチできなかった行は落とす
    if "score" in df.columns:
        df = df.dropna(subset=["score"])

    num_cols = [
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "score",
        "spread_ticks",
        "bid1",
        "ask1",
        "bidqty1",
        "askqty1",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "uptick" not in df.columns and "f1" in df.columns:
        df["uptick"] = pd.to_numeric(df["f1"], errors="coerce").fillna(np.nan)

    cols = [
        "symbol",
        "ts",
        "horizon_sec",
        "ret_bp",
        "label",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "score",
        "spread_ticks",
        "bid1",
        "ask1",
        "bidqty1",
        "askqty1",
        "uptick",
    ]
    df = df[[c for c in cols if c in df.columns]]

    conn.close()
    count = write_csv(out_path, df)
    print(f"[training_set] rows={count} out={out_path}")


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build training CSV from labels_outcome.")
    parser.add_argument("-DatasetId", required=True)
    parser.add_argument("-Out")
    parser.add_argument("-Cols")
    parser.add_argument(
        "-ToleranceSec",
        type=float,
        default=3.0,
        help="asof merge tolerance in seconds (default: 3.0)",
    )
    return parser


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
