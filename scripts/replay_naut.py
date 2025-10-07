#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
replay_naut.py

Offline replay that scans a refeed DB and records strategy events into events_replay.
The generated log captures signal/order/fill/pnl/info events for downstream labelling.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scripts.stream_microbatch import load_json_utf8

JST = timezone(timedelta(hours=9))

EVENTS_REPLAY_DDL = """
CREATE TABLE IF NOT EXISTS events_replay (
  run_id   TEXT NOT NULL,
  symbol   TEXT NOT NULL,
  ts       INTEGER NOT NULL,   -- epoch milliseconds
  etype    TEXT NOT NULL,      -- 'signal','order','fill','pnl','info'
  payload  TEXT NOT NULL,      -- JSON payload
  PRIMARY KEY (run_id, symbol, ts, etype)
);
"""


def _auto_run_id() -> str:
    now = datetime.now(tz=JST)
    return now.strftime("RUN%Y%m%d_%H%M%S")


def _iter_refeed_dbs() -> List[Path]:
    db_dir = Path("db")
    if not db_dir.exists():
        return []
    return sorted(db_dir.glob("naut_market_*_refeed.db"))


def _fetch_dataset_row(db_path: Path, dataset_id: Optional[str]) -> Optional[sqlite3.Row]:
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return None
    try:
        if dataset_id:
            cur = conn.execute(
                "SELECT * FROM dataset_registry WHERE dataset_id=? LIMIT 1",
                (dataset_id,),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM dataset_registry ORDER BY created_at DESC LIMIT 1"
            )
        row = cur.fetchone()
        return row
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def resolve_dataset(dataset_id: Optional[str]) -> Tuple[str, Path, Dict[str, Any]]:
    candidates = _iter_refeed_dbs()
    if not candidates:
        raise SystemExit("ERROR: no refeed DBs found under db/")

    chosen_row: Optional[sqlite3.Row] = None
    chosen_path: Optional[Path] = None

    if dataset_id:
        for db_path in candidates:
            row = _fetch_dataset_row(db_path, dataset_id)
            if row is not None:
                row_dict = dict(row)
                chosen_row = row_dict
                path = Path(row_dict.get("source_db_path") or "")
                if not path:
                    path = db_path
                if not path.exists():
                    path = db_path
                chosen_path = path.resolve()
                break
        if chosen_row is None or chosen_path is None:
            raise SystemExit(f"ERROR: dataset_id {dataset_id} not found in registry.")
    else:
        latest_dt: Optional[datetime] = None
        latest_row: Optional[Dict[str, Any]] = None
        latest_path: Optional[Path] = None
        for db_path in candidates:
            row = _fetch_dataset_row(db_path, None)
            if row is None:
                continue
            row_dict = dict(row)
            created_at = row_dict.get("created_at")
            dt = _parse_iso(created_at) if isinstance(created_at, str) else None
            if dt is None:
                continue
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt
                latest_row = row_dict
                path = Path(row_dict.get("source_db_path") or "")
                if not path:
                    path = db_path
                if not path.exists():
                    path = db_path
                latest_path = path.resolve()
        if latest_row is None or latest_path is None:
            raise SystemExit("ERROR: dataset_registry empty; specify -DatasetId explicitly.")
        chosen_row = latest_row
        chosen_path = latest_path

    dataset_id_final = str(chosen_row.get("dataset_id"))
    info: Dict[str, Any] = {
        "dataset_id": dataset_id_final,
        "created_at": chosen_row.get("created_at"),
        "source_db_path": str(chosen_row.get("source_db_path") or chosen_path),
        "config_json": chosen_row.get("config_json"),
        "code_version": chosen_row.get("code_version"),
    }
    return dataset_id_final, chosen_path, info


def ensure_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(EVENTS_REPLAY_DDL)
    conn.commit()


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mid_price(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


class EventWriter:
    def __init__(self, conn: sqlite3.Connection, run_id: str):
        self.conn = conn
        self.run_id = run_id
        self.offset = int(time.time() * 1000) % 1000
        self.pending = 0
        self.total_rows = 0
        self.last_commit = time.monotonic()

    def log(self, symbol: str, base_ts: float, etype: str, payload: Dict[str, Any]) -> None:
        ts_ms = int(round(max(base_ts, 0.0) * 1000)) + self.offset
        payload_json = json.dumps(payload, ensure_ascii=False)
        while True:
            try:
                self.conn.execute(
                    "INSERT INTO events_replay(run_id,symbol,ts,etype,payload) VALUES (?,?,?,?,?)",
                    (self.run_id, symbol, ts_ms, etype, payload_json),
                )
                break
            except sqlite3.IntegrityError:
                ts_ms += 1
        self.pending += 1
        self.total_rows += 1
        now = time.monotonic()
        if self.pending >= 50 or (now - self.last_commit) >= 1.0:
            self.conn.commit()
            self.pending = 0
            self.last_commit = now

    def flush(self) -> None:
        if self.pending:
            self.conn.commit()
            self.pending = 0
            self.last_commit = time.monotonic()


def _load_symbols(conf: Dict[str, Any]) -> List[str]:
    symbols = conf.get("symbols")
    if isinstance(symbols, Sequence):
        result = [str(sym).strip() for sym in symbols if str(sym).strip()]
        if result:
            return result
    raise SystemExit("ERROR: Config must include non-empty 'symbols' list.")


def _strategy_thresholds(conf: Dict[str, Any]) -> Dict[str, float]:
    naut = conf.get("settings", {}).get("naut", {})
    thresholds = {
        "buy_score": float(naut.get("BUY_SCORE_THR", 6.0)),
        "buy_f1": float(naut.get("BUY_UPTICK_THR", 0.6)),
        "sell_score": float(naut.get("SELL_SCORE_THR", 4.0)),
        "sell_spread": float(naut.get("SELL_SPREAD_MAX", 2)),
        "cooldown": float(naut.get("COOLDOWN_SEC", 5.0)),
        "max_hold": float(naut.get("MAX_HOLD_SEC", 60.0)),
    }
    # guard defaults if config missing
    if thresholds["buy_f1"] <= 0:
        thresholds["buy_f1"] = 0.6
    if thresholds["sell_spread"] <= 0:
        thresholds["sell_spread"] = 2.0
    return thresholds


def _emit_start_info(writer: EventWriter, dataset_id: str, run_id: str, symbols: Sequence[str], info: Dict[str, Any]) -> None:
    payload = {
        "event": "start",
        "dataset_id": dataset_id,
        "run_id": run_id,
        "symbols": list(symbols),
        "created_at": info.get("created_at"),
        "code_version": info.get("code_version", "unknown"),
    }
    writer.log("_meta", time.time(), "info", payload)


def _emit_end_info(writer: EventWriter, dataset_id: str, run_id: str, stats: Dict[str, Any]) -> None:
    stats_payload = dict(stats)
    stats_payload["events_written"] = writer.total_rows + 1  # account for this info row
    payload = {
        "event": "end",
        "dataset_id": dataset_id,
        "run_id": run_id,
        "stats": stats_payload,
        "finished_at": datetime.now(tz=JST).isoformat(timespec="seconds"),
    }
    writer.log("_meta", time.time(), "info", payload)


def replay_events(
    conn: sqlite3.Connection,
    dataset_id: str,
    symbols: Sequence[str],
    thresholds: Dict[str, float],
    writer: EventWriter,
) -> Dict[str, Any]:
    if not symbols:
        return {"rows": 0, "signals": 0, "orders": 0, "fills": 0, "pnl_events": 0, "buys": 0, "sells": 0}

    placeholders = ",".join(["?"] * len(symbols))
    sql = f"""
        SELECT symbol, t_exec, score, f1, spread_ticks, bid1, ask1, bidqty1, askqty1
          FROM features_stream
         WHERE symbol IN ({placeholders})
         ORDER BY t_exec ASC
    """
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, list(symbols))

    stats = {
        "rows": 0,
        "signals": 0,
        "orders": 0,
        "fills": 0,
        "pnl_events": 0,
        "buys": 0,
        "sells": 0,
        "pnl_total": 0.0,
    }

    last_signal_ts: Dict[str, float] = {sym: 0.0 for sym in symbols}
    open_pos: Dict[str, Optional[Dict[str, Any]]] = {sym: None for sym in symbols}

    for row in cur:
        sym = str(row["symbol"])
        ts = _safe_float(row["t_exec"], 0.0) or 0.0
        stats["rows"] += 1

        score = _safe_float(row["score"], 0.0) or 0.0
        f1 = _safe_float(row["f1"], 0.0) or 0.0
        spread = _safe_float(row["spread_ticks"], 0.0) or 0.0
        bid = _safe_float(row["bid1"])
        ask = _safe_float(row["ask1"])
        qty_bid = _safe_float(row["bidqty1"])
        qty_ask = _safe_float(row["askqty1"])
        mid = _mid_price(bid, ask)

        pos = open_pos.get(sym)
        if pos:
            hold_duration = ts - pos["entry_ts"]
            sell_cond = (score <= thresholds["sell_score"]) or (spread >= thresholds["sell_spread"]) or (hold_duration >= thresholds["max_hold"])
            if sell_cond and mid is not None:
                exit_px = bid if bid is not None else mid
                qty = pos["qty"]
                reason = "rule" if hold_duration < thresholds["max_hold"] else "timeout"
                writer.log(
                    sym,
                    ts,
                    "signal",
                    {
                        "side": "SELL",
                        "price": exit_px,
                        "score": score,
                        "spread": spread,
                        "reason": reason,
                        "dataset_id": dataset_id,
                    },
                )
                stats["signals"] += 1
                writer.log(
                    sym,
                    ts,
                    "order",
                    {"side": "SELL", "price": exit_px, "qty": qty, "reason": reason},
                )
                stats["orders"] += 1
                writer.log(
                    sym,
                    ts,
                    "fill",
                    {"side": "SELL", "price": exit_px, "qty": qty, "fill_ts": ts},
                )
                stats["fills"] += 1

                pnl = (exit_px - pos["entry_px"]) * qty
                stats["pnl_total"] += pnl
                writer.log(
                    sym,
                    ts,
                    "pnl",
                    {"side": "SELL", "price": exit_px, "qty": qty, "pnl": pnl},
                )
                stats["pnl_events"] += 1
                stats["sells"] += 1
                open_pos[sym] = None
                last_signal_ts[sym] = ts
                continue

        cooldown_ok = (ts - last_signal_ts.get(sym, 0.0)) >= thresholds["cooldown"]
        buy_cond = (score >= thresholds["buy_score"]) and (f1 >= thresholds["buy_f1"]) and (spread <= thresholds["sell_spread"])
        if not pos and cooldown_ok and buy_cond:
            entry_px = ask if ask is not None else mid
            if entry_px is None:
                continue
            qty = max(qty_ask or 1.0, 1.0)
            writer.log(
                sym,
                ts,
                "signal",
                {
                    "side": "BUY",
                    "price": entry_px,
                    "score": score,
                    "spread": spread,
                    "reason": "rule",
                    "dataset_id": dataset_id,
                },
            )
            stats["signals"] += 1
            writer.log(
                sym,
                ts,
                "order",
                {"side": "BUY", "price": entry_px, "qty": qty, "reason": "rule"},
            )
            stats["orders"] += 1
            writer.log(
                sym,
                ts,
                "fill",
                {"side": "BUY", "price": entry_px, "qty": qty, "fill_ts": ts},
            )
            stats["fills"] += 1
            stats["buys"] += 1
            open_pos[sym] = {"entry_ts": ts, "entry_px": entry_px, "qty": qty}
            last_signal_ts[sym] = ts

    return stats


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay stream features and log events.")
    parser.add_argument("-Config", required=True, help="Path to stream/replay config JSON.")
    parser.add_argument("-RunId", help="Run identifier (default RUN{YYYYMMDD_HHMMSS}).")
    parser.add_argument("-DatasetId", help="Dataset identifier to replay (default latest).")
    parser.add_argument("-Verbose", type=int, default=1, help="Verbosity toggle (reserved).")
    # legacy compatibility arguments (ignored)
    parser.add_argument("-Src", help=argparse.SUPPRESS)
    parser.add_argument("-Dst", help=argparse.SUPPRESS)
    parser.add_argument("-Date", help=argparse.SUPPRESS)
    parser.add_argument("-Symbols", help=argparse.SUPPRESS)
    parser.add_argument("-Speed", help=argparse.SUPPRESS)
    parser.add_argument("-MaxSleep", help=argparse.SUPPRESS)
    parser.add_argument("-LogEvery", help=argparse.SUPPRESS)
    parser.add_argument("-NoSleep", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()

    run_id = args.RunId or _auto_run_id()
    config = load_json_utf8(args.Config)
    symbols = _load_symbols(config)
    thresholds = _strategy_thresholds(config)

    dataset_id, db_path, dataset_info = resolve_dataset(args.DatasetId)

    print("===== Running replay_naut.py =====")
    print(f"[replay] run_id={run_id}")
    print(f"[replay] dataset_id={dataset_id}")
    print(f"[replay] db_path={db_path}")
    print(f"[replay] symbols={symbols}")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    ensure_events_table(conn)
    writer = EventWriter(conn, run_id)
    _emit_start_info(writer, dataset_id, run_id, symbols, dataset_info)

    stats = replay_events(conn, dataset_id, symbols, thresholds, writer)
    _emit_end_info(writer, dataset_id, run_id, stats)
    writer.flush()
    conn.close()

    print(f"[replay] run_id={run_id} wrote {writer.total_rows} rows")


if __name__ == "__main__":
    main()
