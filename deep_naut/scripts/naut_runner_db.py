import sqlite3
from pathlib import Path

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=3000;

CREATE TABLE IF NOT EXISTS features_stream(
  symbol     TEXT    NOT NULL,
  t_exec     INTEGER NOT NULL,
  jst_time   TEXT,
  price_ma3  REAL,
  vol_ma3    REAL,
  imb_ma3    REAL,
  vol_rate   REAL,
  candle_up  REAL,
  PRIMARY KEY(symbol, t_exec)
);

CREATE TABLE IF NOT EXISTS ml_prob(
  symbol   TEXT    NOT NULL,
  t_exec   INTEGER NOT NULL,
  jst_time TEXT,
  prob     REAL    NOT NULL,
  PRIMARY KEY(symbol, t_exec)
);

CREATE TABLE IF NOT EXISTS raw_rest_ticks(
  symbol         TEXT    NOT NULL,
  t_exec         INTEGER NOT NULL,
  jst_time       TEXT,
  last_price     REAL,
  cumulative_vol INTEGER,
  turnover       REAL,
  vwap           REAL,
  payload_json   TEXT,
  PRIMARY KEY(symbol, t_exec)
);

CREATE TABLE IF NOT EXISTS raw_rest_board(
  symbol       TEXT    NOT NULL,
  t_exec       INTEGER NOT NULL,
  jst_time     TEXT,
  best_bid     REAL,
  best_bid_qty INTEGER,
  best_ask     REAL,
  best_ask_qty INTEGER,
  payload_json TEXT,
  PRIMARY KEY(symbol, t_exec)
);
"""


def ensure_features_db(path: str) -> None:
    """
    指定パスのSQLiteを生成し、features_stream / ml_prob / raw_rest_* を初期化する。
    """
    target = Path(path)
    if target.suffix.lower() != ".db" and target.suffix.lower() != ".sqlite":
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    try:
        conn.executescript(DDL)
        conn.commit()
    finally:
        conn.close()
