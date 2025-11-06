# -*- coding: utf-8 -*-

import sqlite3
from typing import Iterator, List, Tuple

EXCLUDE_COLS = {
    "id",
    "rowid",
    "_rowid_",
    "symbol",
    "t",
    "t_ms",
    "t_exec",
    "ts",
    "ts_ms",
}


def _numeric_feature_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    # PRAGMA でカラム一覧→数値系だけ採用（INTEGER/REAL）
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = []
    for _, name, ctype, *_ in cur.fetchall():
        n = name.lower()
        if n in EXCLUDE_COLS:
            continue
        ct = (ctype or "").upper()
        if "INT" in ct or "REAL" in ct or "NUM" in ct or ct == "":
            cols.append(name)
    if not cols:
        raise RuntimeError(f"No numeric feature columns detected in table '{table}'.")
    return cols


def iter_features_from_db(
    db_path: str,
    table: str,
    symbol: str,
    since_exec: float | None = None,
    batch: int = 512,
) -> Tuple[Iterator[List[float]], List[str]]:
    """
    指定DBから features を順序どおりに逐次取得するジェネレータ。
    返り値: (イテレータ, 採用カラム名リスト)
    - since_exec: t_exec（秒 or ms いずれでも可）より後だけ欲しい場合
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    feat_cols = _numeric_feature_cols(conn, table)
    # t_exec 相当のカラムを推定（存在チェック）
    time_cols = [
        c for c in ["t_exec", "ts_ms", "ts", "t_ms", "t"] if _has_col(conn, table, c)
    ]
    order_col = time_cols[0] if time_cols else "rowid"

    # since 条件
    where = "symbol=?"
    params = [symbol]
    if since_exec is not None and _has_col(conn, table, order_col):
        where += f" AND {order_col} > ?"
        params.append(since_exec)

    cur = conn.execute(
        f"SELECT {','.join(['symbol']+feat_cols)} FROM {table} "
        f"WHERE {where} ORDER BY {order_col} ASC",
        params,
    )

    def _row_iter():
        for r in cur:
            yield [float(r[c]) if r[c] is not None else 0.0 for c in feat_cols]
        conn.close()

    return _row_iter(), feat_cols


def _has_col(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1].lower() == col.lower() for row in cur.fetchall())
