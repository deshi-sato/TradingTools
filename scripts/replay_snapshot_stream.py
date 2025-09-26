#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replay tick batches from rss_snapshot.db into features_stream to mimic a PUSH feed.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import time
import shlex
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scripts.common_config import load_json_utf8
from scripts.feature_calc import depth_imbalance, spread_bp, uptick_ratio
from scripts.stream_microbatch import ensure_tables, insert_features


# ----------------------------
# helpers
# ----------------------------
def parse_csv_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def detect_latest_day(conn: sqlite3.Connection, symbols: Sequence[str]) -> Optional[str]:
    # prefer tick_batch
    base = ["SELECT DISTINCT substr(ts_window_start, 1, 10) AS day FROM tick_batch"]
    params: List[str] = []
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        base.append(f"WHERE ticker IN ({placeholders})")
        params.extend(symbols)
    base.append("ORDER BY day DESC LIMIT 1")
    row = conn.execute(" ".join(base), params).fetchone()
    if row and row[0]:
        return row[0]

    # fallback: today_data
    base = ["SELECT DISTINCT substr(datetime, 1, 10) AS day FROM today_data"]
    params = []
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        base.append(f"WHERE ticker IN ({placeholders})")
        params.extend(symbols)
    base.append("ORDER BY day DESC LIMIT 1")
    row = conn.execute(" ".join(base), params).fetchone()
    if row and row[0]:
        return row[0]
    return None


def fetch_tick_batches(
    conn: sqlite3.Connection,
    day: str,
    symbols: Sequence[str],
    limit: int,
) -> List[Dict[str, object]]:
    clauses = [
        "SELECT ticker, ts_window_start, ts_window_end, ticks, upticks, downticks, vol_sum, last_price",
        "FROM tick_batch",
        "WHERE substr(ts_window_start, 1, 10)=?",
    ]
    params: List[object] = [day]
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        clauses.append(f"AND ticker IN ({placeholders})")
        params.extend(symbols)
    clauses.append("ORDER BY ts_window_end")
    if limit > 0:
        clauses.append("LIMIT ?")
        params.append(int(limit))
    cur = conn.execute(" ".join(clauses), params)
    return [dict(row) for row in cur.fetchall()]


def fetch_today_rows(
    conn: sqlite3.Connection,
    day: str,
    symbols: Sequence[str],
    limit: int,
) -> List[Dict[str, object]]:
    clauses = [
        "SELECT ticker, datetime, open, high, low, close, volume",
        "FROM today_data",
        "WHERE substr(datetime, 1, 10)=?",
    ]
    params: List[object] = [day]
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        clauses.append(f"AND ticker IN ({placeholders})")
        params.extend(symbols)
    clauses.append("ORDER BY datetime")
    if limit > 0:
        clauses.append("LIMIT ?")
        params.append(int(limit))
    cur = conn.execute(" ".join(clauses), params)

    rows: List[Dict[str, object]] = []
    for row in cur.fetchall():
        rec = dict(row)
        ticker = str(rec.get("ticker") or "").strip()
        ts_iso = rec.get("datetime")
        open_price = rec.get("open")
        close_price = rec.get("close")
        volume = rec.get("volume")
        upticks = 0
        downticks = 0
        if open_price is not None and close_price is not None:
            if close_price >= open_price:
                upticks = 1
            else:
                downticks = 1
        rows.append(
            {
                "ticker": ticker,
                "ts_window_start": ts_iso,
                "ts_window_end": ts_iso,
                "ticks": 1,
                "upticks": upticks,
                "downticks": downticks,
                "vol_sum": volume or 0,
                "last_price": close_price,
            }
        )
    return rows


def lookup_orderbook(
    conn: sqlite3.Connection,
    ticker: str,
    ts_iso: Optional[str],
) -> Dict[str, object]:
    if not ticker or not ts_iso:
        return {}
    row = conn.execute(
        "SELECT bid1, ask1, spread_bp, buy_top3, sell_top3 FROM orderbook_snapshot "
        "WHERE ticker=? AND ts <= ? ORDER BY ts DESC LIMIT 1",
        (ticker, ts_iso),
    ).fetchone()
    return dict(row) if row else {}


def build_feature_row(
    batch: Dict[str, object],
    orderbook: Dict[str, object],
) -> Dict[str, object]:
    ticker = batch.get("ticker")
    ts_iso = batch.get("ts_window_end") or batch.get("ts_window_start")
    if not ts_iso:
        ts_iso = datetime.now().isoformat(timespec="seconds")

    upticks = int(batch.get("upticks") or 0)
    downticks = int(batch.get("downticks") or 0)
    ratio = float(uptick_ratio(upticks, downticks))

    vol_sum_val = batch.get("vol_sum")
    try:
        vol_sum = float(vol_sum_val)
    except (TypeError, ValueError):
        vol_sum = 0.0

    buy3 = 0
    sell3 = 0
    spread_val = None
    if orderbook:
        buy3 = int(orderbook.get("buy_top3") or 0)
        sell3 = int(orderbook.get("sell_top3") or 0)
        spread_val = orderbook.get("spread_bp")
        if spread_val is None:
            spread_val = spread_bp(orderbook.get("bid1"), orderbook.get("ask1"))

    depth = float(depth_imbalance(buy3, sell3))

    return {
        "ticker": ticker,
        "ts": ts_iso,
        "uptick_ratio": ratio,
        "vol_sum": vol_sum,
        "spread_bp": spread_val,
        "buy_top3": buy3,
        "sell_top3": sell3,
        "depth_imbalance": depth,
        "burst_buy": 0,
        "burst_sell": 0,
        "burst_score": 0.0,
        "streak_len": 0,
        "surge_vol_ratio": 1.0,
        "last_signal_ts": "",
    }


def stream_batches(
    batches: List[Dict[str, object]],
    source_conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
    *,
    speed: float,
    no_sleep: bool,
    max_sleep: float,
    verbose: bool,
) -> None:
    total = len(batches)
    if total == 0:
        return

    start_data: Optional[datetime] = None
    start_real: Optional[float] = None

    for idx, batch in enumerate(batches, 1):
        ts_iso = batch.get("ts_window_end") or batch.get("ts_window_start")
        data_time = parse_iso8601(ts_iso)

        if start_data is None and data_time is not None:
            start_data = data_time
            start_real = time.monotonic()

        # pace control
        if (
            not no_sleep
            and speed > 0
            and data_time is not None
            and start_data is not None
            and start_real is not None
        ):
            data_elapsed = (data_time - start_data).total_seconds()
            target_elapsed = data_elapsed / speed
            real_elapsed = time.monotonic() - start_real
            sleep_for = target_elapsed - real_elapsed
            if sleep_for > 0:
                time.sleep(min(sleep_for, max(0.0, max_sleep)))

        # enrich & insert
        orderbook = lookup_orderbook(source_conn, str(batch.get("ticker") or ""), ts_iso)
        feature = build_feature_row(batch, orderbook)
        insert_features(target_conn, [feature])

        if verbose:
            ratio = feature["uptick_ratio"]
            vol_sum = feature["vol_sum"]
            spread_val = feature["spread_bp"]
            spread_txt = f"{spread_val:.2f}" if isinstance(spread_val, (int, float)) else "None"
            print(
                f"[REPLAY] {idx:05d}/{total:05d} {feature['ticker']} ts={feature['ts']} "
                f"ratio={ratio:.2f} vol={vol_sum:.0f} spread={spread_txt}"
            )


def launch_naut_runner(naut_config: str, naut_extra=None) -> subprocess.Popen:
    """
    Launch 'python -m scripts.naut_runner -Config <json> [extra ...]'
    naut_extra: list or str (e.g., ['--print-summary'] or '--print-summary')
    """
    args = [sys.executable, "-m", "scripts.naut_runner", "-Config", naut_config]
    if naut_extra:
        if isinstance(naut_extra, str):
            args.extend(shlex.split(naut_extra))
        else:
            args.extend(naut_extra)
    print(f"[NAUT] launch: {' '.join(args)}")
    return subprocess.Popen(args)


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay historical tick data from rss_snapshot.db to features_stream.",
    )
    parser.add_argument("-Config", help="Optional config JSON (e.g. config/stream_settings.json).")
    parser.add_argument("--source-db", help="Path to snapshot DB (default: from config or rss_snapshot.db).")
    parser.add_argument("--target-db", help="Destination DB for features_stream (default: source DB).")
    parser.add_argument("--date", help="YYYY-MM-DD to replay (default: latest day).")
    parser.add_argument("--symbols", help="Comma separated tickers.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0=realtime).")
    parser.add_argument("--no-sleep", action="store_true", help="Disable waits between batches.")
    parser.add_argument("--max-sleep", type=float, default=2.5, help="Cap on per-batch sleep seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Max batches (0=no limit).")
    parser.add_argument("--truncate", action="store_true", help="DELETE target day rows before replay.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    parser.add_argument("--run-naut", action="store_true", help="Launch naut_runner while replaying.")
    parser.add_argument("--naut-config", help="Config JSON for naut_runner (default: same as -Config).")
    # 可変長にして PowerShell でも安全に受け取れるようにする
    parser.add_argument(
        "--naut-extra",
        nargs="*",
        default=None,
        help="Extra args for naut_runner (e.g. --print-summary)",
    )
    parser.add_argument("--naut-grace", type=float, default=3.0, help="Seconds to wait before terminating naut_runner.")
    parser.add_argument("--leave-naut", action="store_true", help="Leave naut_runner running after replay.")
    return parser.parse_args()


# ----------------------------
# main
# ----------------------------
def main() -> None:
    args = parse_args()

    # load config (optional)
    config_payload: Dict[str, object] = {}
    config_path: Optional[Path] = None
    if args.Config:
        config_path = Path(args.Config).expanduser()
        if not config_path.exists():
            raise SystemExit(f"[ERROR] Config file not found: {config_path}")
        config_payload = load_json_utf8(str(config_path))

    # decide DBs
    source_db = args.source_db or config_payload.get("snapshot_db") or config_payload.get("db_path") or "rss_snapshot.db"
    target_db = args.target_db or config_payload.get("features_db") or config_payload.get("judge_db") or config_payload.get("db_path") or source_db

    source_path = Path(str(source_db)).expanduser()
    target_path = Path(str(target_db)).expanduser()
    if not source_path.exists():
        raise SystemExit(f"[ERROR] Source DB not found: {source_path}")

    ensure_tables(str(target_path))

    # symbols
    symbols = parse_csv_list(args.symbols)
    if not symbols:
        config_symbols = config_payload.get("symbols")
        if isinstance(config_symbols, list):
            symbols = [str(sym).strip() for sym in config_symbols if str(sym).strip()]

    # open DBs
    source_conn = sqlite3.connect(str(source_path))
    source_conn.row_factory = sqlite3.Row
    target_conn = sqlite3.connect(str(target_path))
    target_conn.row_factory = sqlite3.Row

    naut_proc: Optional[subprocess.Popen] = None
    try:
        # determine day
        day = args.date or detect_latest_day(source_conn, symbols)
        if not day:
            raise SystemExit("[ERROR] Could not determine a trading day to replay.")

        # truncate
        if args.truncate:
            target_conn.execute("DELETE FROM features_stream WHERE substr(ts, 1, 10)=?", (day,))
            target_conn.commit()
            if not args.quiet:
                print(f"[INIT] cleared features_stream rows for {day}")

        # info
        if not args.quiet:
            print(f"[INFO] replay day={day} source={source_path} target={target_path}")
            if symbols:
                print(f"[INFO] symbols={symbols}")

        # fetch rows
        batches = fetch_tick_batches(source_conn, day, symbols, args.limit)
        if not batches:
            if not args.quiet:
                print("[WARN] tick_batch had no rows; falling back to today_data")
            batches = fetch_today_rows(source_conn, day, symbols, args.limit)
        if not batches:
            raise SystemExit("[ERROR] No data rows found to replay.")

        if not args.quiet:
            per_symbol: Dict[str, int] = {}
            for b in batches:
                k = str(b.get("ticker") or "")
                per_symbol[k] = per_symbol.get(k, 0) + 1
            print(f"[INFO] total batches={len(batches)} per_symbol={per_symbol}")

        # launch naut_runner if requested
        if args.run_naut:
            naut_conf = args.naut_config or args.Config or "config/stream_settings.json"
            naut_path = Path(str(naut_conf)).expanduser()
            if not naut_path.exists():
                raise SystemExit(f"[ERROR] naut_runner config not found: {naut_path}")
            naut_proc = launch_naut_runner(str(naut_path), args.naut_extra)

        # stream
        try:
            stream_batches(
                batches,
                source_conn,
                target_conn,
                speed=args.speed,
                no_sleep=args.no_sleep,
                max_sleep=max(0.0, args.max_sleep),
                verbose=not args.quiet,
            )
        except KeyboardInterrupt:
            print("\n[INFO] replay interrupted by user")
        finally:
            if naut_proc is not None:
                if args.leave_naut:
                    print(f"[NAUT] leaving process running (pid={naut_proc.pid})")
                else:
                    if args.naut_grace > 0:
                        time.sleep(args.naut_grace)
                    naut_proc.terminate()
                    try:
                        naut_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        naut_proc.kill()
                        naut_proc.wait(timeout=5)
                    print("[NAUT] stopped")
    finally:
        source_conn.close()
        target_conn.close()


if __name__ == "__main__":
    main()
