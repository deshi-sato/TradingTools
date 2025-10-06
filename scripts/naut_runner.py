#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
naut_runner.py

PAPER mode consumer for stream_microbatch features_stream.
- Polls naut_market.db.features_stream for latest f1..f6 + score entries.
- Emits paper BUY/SELL decisions based on score-driven rules.
- Persists pseudo orders into naut_ops.db.orders_log and paper_pairs for audit.
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
import atexit
import ctypes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from scripts.common_config import load_json_utf8

logger = logging.getLogger(__name__)

_singleton_handle = None
_pidfile_path: Optional[Path] = None


def _cleanup_pid():
    global _singleton_handle, _pidfile_path
    try:
        if _singleton_handle:
            ctypes.windll.kernel32.CloseHandle(_singleton_handle)
    except Exception:
        pass
    try:
        if _pidfile_path and _pidfile_path.exists():
            _pidfile_path.unlink()
    except Exception:
        pass


def singleton_guard(tag: str) -> None:
    """Prevent duplicate launcher instances via named mutex + pid file."""
    global _singleton_handle, _pidfile_path
    name = f"Global\{tag}"
    _singleton_handle = ctypes.windll.kernel32.CreateMutexW(None, False, name)
    if ctypes.GetLastError() == 183:
        print(f"[ERROR] {tag} already running", file=sys.stderr)
        sys.exit(1)
    pid_dir = Path("runtime/pids")
    pid_dir.mkdir(parents=True, exist_ok=True)
    _pidfile_path = pid_dir / f"{tag}.pid"
    try:
        _pidfile_path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass
    atexit.register(_cleanup_pid)


class FeaturePoller:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def fetch_recent(self, symbol: str, last_ts: float, limit: int = 200) -> List[Dict]:
        cur = self.conn.execute(
            """
SELECT symbol, t_exec, f1, f2, f3, f4, f5, f6,
       score, spread_ticks, bid1, ask1, bidqty1, askqty1
FROM features_stream
WHERE symbol=? AND t_exec>? ORDER BY t_exec ASC LIMIT ?
""",
            (symbol, last_ts, limit),
        )
        columns = [col[0] for col in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


ORDERS_LOG_DDL = """
CREATE TABLE IF NOT EXISTS orders_log(
  ts REAL NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  type TEXT NOT NULL,
  qty REAL,
  price REAL,
  action TEXT,
  status TEXT,
  meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_log_symbol_ts ON orders_log(symbol, ts);
"""

PAPER_PAIRS_DDL = """
CREATE TABLE IF NOT EXISTS paper_pairs(
  pair_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol      TEXT NOT NULL,
  entry_ts    REAL NOT NULL,
  entry_px    REAL NOT NULL,
  exit_ts     REAL,
  exit_px     REAL,
  pnl         REAL
);
CREATE INDEX IF NOT EXISTS idx_pairs_sym_time ON paper_pairs(symbol, entry_ts);
"""



def seed_last_ts(conn: sqlite3.Connection, symbols: List[str], back: float = 2.0) -> Dict[str, float]:
    seed: Dict[str, float] = {}
    for sym in symbols:
        try:
            row = conn.execute("SELECT IFNULL(MAX(t_exec),0) FROM features_stream WHERE symbol=?", (sym,)).fetchone()
            max_ts = float(row[0]) if row and row[0] is not None else 0.0
        except Exception:
            max_ts = 0.0
        seed[sym] = max(0.0, max_ts - back)
    return seed

def ensure_ops_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(ORDERS_LOG_DDL)
    conn.executescript(PAPER_PAIRS_DDL)


def to_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def paper_entry_price_qty(row: Dict) -> Optional[tuple[float, int]]:
    ask = to_float(row.get("ask1"))
    if ask is None:
        return None
    qty = max(1, to_int(row.get("askqty1"), 1))
    return ask + 1.0, qty


def paper_exit_price_qty(row: Dict) -> Optional[tuple[float, int]]:
    bid = to_float(row.get("bid1"))
    if bid is None:
        return None
    qty = max(1, to_int(row.get("bidqty1"), 1))
    return bid - 1.0, qty


def run_paper_mode(conf: Dict[str, object]) -> None:
    symbols: List[str] = [str(s) for s in conf.get("symbols", []) if str(s)]
    if not symbols:
        raise SystemExit("ERROR: symbols empty")

    db_path = str(conf.get("db_path", "db/naut_market.db"))
    poll_interval = float(conf.get("runner_poll_sec", 0.25))
    poll_limit = int(conf.get("runner_batch_limit", 200))
    COOLDOWN_SEC = float(conf.get("runner_cooldown_sec", 5.0))
    MAX_HOLD = float(conf.get("runner_max_hold_sec", 20.0))
    TRADING_START = str(conf.get("trading_start", "09:00"))
    TRADING_END = str(conf.get("trading_end", "15:30"))

    orders_db_path = str(conf.get("orders_db_path", "db/naut_ops.db"))
    Path(orders_db_path).parent.mkdir(parents=True, exist_ok=True)

    poller = FeaturePoller(db_path)
    conn_mkt = poller.conn
    conn_ops = sqlite3.connect(orders_db_path, check_same_thread=False, isolation_level=None)
    conn_ops.execute("PRAGMA journal_mode=WAL;")
    conn_ops.execute("PRAGMA synchronous=NORMAL;")
    ensure_ops_tables(conn_ops)

    last_ts = seed_last_ts(conn_mkt, symbols)
    last_fire: Dict[str, float] = {sym: 0.0 for sym in symbols}
    open_pos: Dict[str, Optional[Dict[str, float]]] = {sym: None for sym in symbols}

    stats = {"polled": 0, "trig": 0, "buy": 0, "sell": 0}
    last_stats_log = time.time()

    def in_trading_hours() -> bool:
        now_str = datetime.now().strftime("%H:%M")
        return TRADING_START <= now_str <= TRADING_END

    logger.info(
        "paper runner start symbols=%s db=%s orders_db=%s poll=%.3fs limit=%d",
        symbols,
        db_path,
        orders_db_path,
        poll_interval,
        poll_limit,
    )

    try:
        while True:
            if not in_trading_hours():
                logger.info("Market closed: skip trading cycle")
                time.sleep(30)
                continue

            for sym in symbols:
                rows = poller.fetch_recent(sym, last_ts[sym], poll_limit)
                stats["polled"] += len(rows)

                if rows:
                    last_ts[sym] = rows[-1]["t_exec"]
                    r = rows[-1]
                    sp_val = r.get("spread_ticks", -1)
                    try:
                        sp_fmt = int(float(sp_val))
                    except (TypeError, ValueError):
                        sp_fmt = -1
                    logger.info(
                        "beat %s last_ts=%.3f rows=%d score=%.2f f1=%.2f sp=%d",
                        sym,
                        last_ts[sym],
                        len(rows),
                        float(r.get("score", -1.0)),
                        float(r.get("f1", -1.0)),
                        sp_fmt,
                    )
                else:
                    logger.info("beat %s last_ts=%.3f rows=0", sym, last_ts[sym])
                    continue

                for row in rows:
                    now = time.time()
                    sym = row["symbol"]
                    pos = open_pos.get(sym)
                    try:
                        score = float(row.get("score", -1))
                    except (TypeError, ValueError):
                        score = -1.0
                    try:
                        f1 = float(row.get("f1", -1))
                    except (TypeError, ValueError):
                        f1 = -1.0
                    try:
                        sp = int(float(row.get("spread_ticks", -1)))
                    except (TypeError, ValueError):
                        sp = -1

                    sell_cond = (score <= 4.0) or (sp >= 2)
                    buy_cond = (score >= 6.0) and (f1 >= 0.6)

                    if pos:
                        hold_duration = now - pos["ts"]
                        if sell_cond or hold_duration >= MAX_HOLD:
                            exit_info = paper_exit_price_qty(row)
                            if exit_info is None:
                                continue
                            px, qty = exit_info
                            stats["trig"] += 1
                            stats["sell"] += 1
                            reason = "rule" if sell_cond else "timeout"
                            meta = json.dumps({"score": score, "f1": f1, "sp": sp, "reason": reason}, ensure_ascii=False)
                            conn_ops.execute(
                                "INSERT INTO orders_log(ts,symbol,side,type,qty,price,action,status,meta) VALUES(?,?,?,?,?,?,?,?,?)",
                                (now, sym, "SELL", "paper", qty, px, "place", "ok", meta),
                            )
                            pnl = px - pos["px"]
                            conn_ops.execute(
                                "UPDATE paper_pairs SET exit_ts=?, exit_px=?, pnl=? WHERE pair_id=?",
                                (now, px, pnl, pos["pair_id"]),
                            )
                            logger.info(
                                "SELL %s px=%.1f pnl=%.1f reason=%s pair=%d",
                                sym,
                                px,
                                pnl,
                                reason,
                                pos["pair_id"],
                            )
                            open_pos[sym] = None
                        continue

                    if (now - last_fire[sym] >= COOLDOWN_SEC) and buy_cond:
                        entry_info = paper_entry_price_qty(row)
                        if entry_info is None:
                            continue
                        px, qty = entry_info
                        stats["trig"] += 1
                        stats["buy"] += 1
                        meta = json.dumps({"score": score, "f1": f1, "sp": sp}, ensure_ascii=False)
                        conn_ops.execute(
                            "INSERT INTO orders_log(ts,symbol,side,type,qty,price,action,status,meta) VALUES(?,?,?,?,?,?,?,?,?)",
                            (now, sym, "BUY", "paper", qty, px, "place", "ok", meta),
                        )
                        pair_id = conn_ops.execute(
                            "INSERT INTO paper_pairs(symbol,entry_ts,entry_px) VALUES(?,?,?)",
                            (sym, now, px),
                        ).lastrowid
                        open_pos[sym] = {"ts": now, "px": px, "pair_id": pair_id}
                        last_fire[sym] = now
                        logger.info(
                            "BUY %s px=%.1f score=%.2f f1=%.2f sp=%d pair=%d",
                            sym,
                            px,
                            score,
                            f1,
                            sp,
                            pair_id,
                        )
                        continue

            if time.time() - last_stats_log >= 5.0:
                logger.info(
                    "stats: polled=%d trig=%d buy=%d sell=%d",
                    stats["polled"],
                    stats["trig"],
                    stats["buy"],
                    stats["sell"],
                )
                stats = {"polled": 0, "trig": 0, "buy": 0, "sell": 0}
                last_stats_log = time.time()

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("paper runner stop: KeyboardInterrupt")
    finally:
        try:
            conn_ops.close()
        except Exception:
            pass
        try:
            poller.conn.close()
        except Exception:
            pass


def main() -> None:
    singleton_guard("naut_runner")
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Mode", default="paper")
    ap.add_argument("-Verbose", type=int, default=1)
    args = ap.parse_args()

    conf = load_json_utf8(args.Config)

    log_path = "logs/naut_runner_paper.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    mode = str(args.Mode).lower()
    if mode != "paper":
        raise SystemExit(f"ERROR: Mode {args.Mode} not supported")

    run_paper_mode(conf)


if __name__ == "__main__":
    main()
