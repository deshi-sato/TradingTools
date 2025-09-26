# scripts/naut_runner.py

import argparse
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from scripts.burst_helper import burst_bonus
from scripts.common_config import load_json_utf8

logger = logging.getLogger(__name__)

# ------------------------ base score / logging helpers ------------------------

def compute_base_score(side: str, f: dict, cfg: dict, vol_min_by_code: dict) -> float:
    vol_min = max(1, int(vol_min_by_code.get(f["ticker"], cfg["VOL_MIN"])))
    vol_over = max(0.0, (f["vol_sum"] / vol_min) - 1.0)
    if side == "BUY":
        u = float(f.get("uptick_ratio", 0.0))
        i = max(0.0, float(f.get("depth_imbalance", 0.0)))
        return min(1.0, 0.80 * u + 0.15 * i + 0.05 * vol_over)
    else:  # SELL
        u_rev = 1.0 - float(f.get("uptick_ratio", 1.0))
        i = max(0.0, -float(f.get("depth_imbalance", 0.0)))
        return min(1.0, 0.80 * u_rev + 0.15 * i + 0.05 * vol_over)

def log_side(side: str, f: dict, r: dict) -> None:
    spr = f["spread_bp"]
    spr_txt = (None if spr is None else f"{spr:.1f}")
    logger.info(
        f"[{side}] {f['ticker']} ts={f['ts']} "
        f"uptick={f['uptick_ratio']:.2f} imb={f['depth_imbalance']:.2f} "
        f"spr={spr_txt} vol={f['vol_sum']:.0f} | "
        f"base={r['base']:.3f} adj={r['adjusted']:.3f} bonus={r['bonus']:.3f} thr={r['threshold']:.2f}"
    )

def apply_burst_and_decide(
    conn,
    ticker: str,
    base_score: float,
    threshold: float = 0.60,
    *,
    since: str,
    until: str,
):
    adjusted, bonus, ts = burst_bonus(
        conn, ticker, base_score,
        k=0.30, tau_sec=12.0,
        lookback_sec=8.0,
        min_gate=0.20,
    )

    cooldown_sec = 5.0
    cur = conn.cursor()
    cur.execute(
        """SELECT ts FROM features_stream
           WHERE ticker=? AND ts >= ? AND ts <= ? AND (burst_buy=1 OR burst_sell=1)
           ORDER BY ts DESC LIMIT 1""",
        (ticker, since, until),
    )
    last = cur.fetchone()
    cooled = True
    if last:
        from datetime import timezone
        dt = datetime.fromisoformat(last[0].replace("Z", ""))
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        age = (datetime.now(timezone.utc).replace(tzinfo=None) - dt).total_seconds()
        cooled = (age >= cooldown_sec)

    take = (adjusted >= threshold) and cooled
    return {
        "ticker": ticker,
        "base": base_score,
        "adjusted": adjusted,
        "bonus": bonus,
        "burst_ts": ts,
        "threshold": threshold,
        "decision": bool(take),
        "cooled": cooled,
    }

def log_decision(r: dict) -> None:
    logger.debug(
        "burst_decision | %(ticker)s | base=%(base).3f -> adj=%(adjusted).3f (+%(bonus).3f) "
        "thr=%(threshold).2f cooled=%(cooled)s decision=%(decision)s ts=%(burst_ts)s",
        r,
    )

def _summarize_counts(per: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for symbol, counts in per.items():
        filtered = {side: value for side, value in counts.items() if value}
        if filtered:
            summary[symbol] = filtered
    return summary

def _debug_print_counts(label: str, total: int, per: Dict[str, Dict[str, int]], enabled: bool) -> None:
    if not enabled:
        return
    summary = _summarize_counts(per)
    print(f"[DBG] {label}: total={total}  per={summary}", flush=True)

# ----------------------------- utility functions ------------------------------

def within_window(spec: Optional[str]) -> bool:
    if not spec:
        return True
    try:
        s, e = spec.split("-")
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        now = datetime.now().time()
        return dtime(sh, sm) <= now <= dtime(eh, em)
    except Exception:
        return True

# --------------------------------- profiles -----------------------------------

PROD = dict(
    BUY=dict(uptick=0.62, imb=0.15, spread=3.0),
    SELL=dict(uptick=0.38, imb=-0.15, spread=3.0),
    VOL_MIN=600,
    WINDOW=5,
    ALLOW_SPREAD_NONE=False,
)

DEMO = dict(
    BUY=dict(uptick=0.40, imb=-1.0, spread=9999.0),
    SELL=dict(uptick=0.60, imb=1.0, spread=9999.0),
    VOL_MIN=0,
    WINDOW=4,
    ALLOW_SPREAD_NONE=True,
)

PRODLITE = dict(
    BUY=dict(uptick=0.65, imb=0.00, spread=5.0),
    SELL=dict(uptick=0.35, imb=-0.00, spread=5.0),
    VOL_MIN=1200,
    WINDOW=4,  # ←検証用。運用では4に戻す
    ALLOW_SPREAD_NONE=False,
)

RECENT_LIMIT = 40

# ------------------------------- db poller ------------------------------------

class FeaturePoller:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.recent_dirbuf: Dict[str, List[int]] = {}

    def fetch_recent(self, symbol: str, since: str, until: str, limit: int = RECENT_LIMIT) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT * FROM features_stream WHERE ticker=? AND ts >= ? AND ts <= ? ORDER BY ts DESC LIMIT ?",
            (symbol, since, until, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def record_direction(self, symbol: str, want: int, window_count: int) -> None:
        buf = self.recent_dirbuf.setdefault(symbol, [])
        buf.append(want)
        if len(buf) > window_count:
            buf.pop(0)

    def consistent(self, symbol: str, want: int, window_count: int) -> bool:
        buf = self.recent_dirbuf.get(symbol, [])
        if len(buf) < window_count:
            return False
        agree = sum(1 for v in buf if v == want)
        return agree >= (window_count // 2 + 1)

# ------------------------------- signal checks -------------------------------

def spread_ok(x: Optional[float], limit: float, allow_none: bool) -> bool:
    if x is None:
        return allow_none
    return x <= limit

def is_buy_signal(
    f: Dict,
    vol_min_by_code: Dict[str, int],
    *,
    buy_uptick_thr: float,
    buy_require_imb: bool,
    buy_vol_min: int,
    buy_spread_max: float,
    skips: Optional[Dict[str, int]] = None,
    skips_detail: Optional[Dict[str, int]] = None,
) -> bool:
    ticker = f.get("ticker")

    vol_override = vol_min_by_code.get(ticker)
    try:
        vol_override_int = int(vol_override) if vol_override is not None else None
    except (TypeError, ValueError):
        vol_override_int = None
    effective_vol_min = max(0, buy_vol_min)
    if vol_override_int is not None:
        effective_vol_min = max(effective_vol_min, vol_override_int)

    vol_sum_raw = f.get("vol_sum")
    try:
        vol_sum_val = float(vol_sum_raw)
    except (TypeError, ValueError):
        vol_sum_val = None

    spread_raw = f.get("spread_bp")
    try:
        spread_val = float(spread_raw) if spread_raw is not None else None
    except (TypeError, ValueError):
        spread_val = None

    uptick_raw = f.get("uptick_ratio")
    try:
        uptick_val = float(uptick_raw)
    except (TypeError, ValueError):
        uptick_val = None

    imb_raw = f.get("depth_imbalance")
    try:
        imb_val = float(imb_raw)
    except (TypeError, ValueError):
        imb_val = None

    cond_uptick = (uptick_val is not None) and (uptick_val >= buy_uptick_thr)
    cond_vol = (vol_sum_val is not None) and (vol_sum_val > effective_vol_min)
    cond_spread = (spread_val is not None) and (spread_val <= buy_spread_max)
    cond_imb = (imb_val is not None) and (imb_val > 0.0)

    if not (cond_uptick and cond_vol and cond_spread):
        if skips_detail is not None:
            if not cond_uptick:
                skips_detail['uptick'] += 1
            if not cond_vol:
                skips_detail['vol'] += 1
            if not cond_spread:
                skips_detail['spread'] += 1
        if skips is not None:
            skips['base'] += 1
        return False
    if buy_require_imb and not cond_imb:
        if skips_detail is not None: skips_detail['imb'] += 1
        if skips is not None:
            skips['base'] += 1
        return False

    return True

def is_sell_signal(
    f: Dict,
    cfg: Dict,
    vol_min_by_code: Dict[str, int],
    *,
    sell_uptick_thr: float,
    sell_require_imb: bool,
    skips: Optional[Dict[str, int]] = None,
    skips_detail: Optional[Dict[str, int]] = None,
) -> bool:
    vol_min = vol_min_by_code.get(f["ticker"], cfg["VOL_MIN"])
    if f["vol_sum"] < vol_min:
        if skips_detail is not None: skips_detail['vol'] += 1
        if skips is not None:
            skips['base'] += 1
        return False
    if not spread_ok(f["spread_bp"], cfg["SELL"]["spread"], cfg["ALLOW_SPREAD_NONE"]):
        if skips_detail is not None: skips_detail['spread'] += 1
        if skips is not None:
            skips['base'] += 1
        return False

    uptick = f.get("uptick_ratio")
    imb = f.get("depth_imbalance")

    sell_gate = (uptick is not None) and (uptick <= (1.0 - sell_uptick_thr))
    cond_imb = (imb is not None) and (imb < 0)
    cond_core = (sell_gate and cond_imb) if sell_require_imb else sell_gate
    if not cond_core:
        if skips_detail is not None:
            if not sell_gate:
                skips_detail['uptick'] += 1
            if sell_require_imb and not cond_imb:
                skips_detail['imb'] += 1
        if skips is not None:
            skips['base'] += 1
        return False

    return True

# ------------------------------- main loop ------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    ap.add_argument("--print-summary", action="store_true")
    ap.add_argument("--profile", choices=["prod", "prodlite", "demo"], default="prod")
    args = ap.parse_args()
    print_summary = args.print_summary

    conf = load_json_utf8(args.Config)
    settings_naut = {}
    settings_root = conf.get("settings")
    if isinstance(settings_root, dict):
        candidate = settings_root.get("naut")
        if isinstance(candidate, dict):
            settings_naut = candidate

    sell_uptick_raw = settings_naut.get("SELL_UPTICK_THR", 0.50)
    try:
        sell_uptick_thr = float(sell_uptick_raw)
    except (TypeError, ValueError):
        sell_uptick_thr = 0.50

    sell_require_raw = settings_naut.get("SELL_REQUIRE_IMB", True)
    if sell_require_raw is None:
        sell_require_imb = True
    elif isinstance(sell_require_raw, str):
        sell_require_imb = sell_require_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        sell_require_imb = bool(sell_require_raw)

    buy_uptick_raw = settings_naut.get("BUY_UPTICK_THR", 0.65)
    try:
        buy_uptick_thr = float(buy_uptick_raw)
    except (TypeError, ValueError):
        buy_uptick_thr = 0.65

    buy_require_raw = settings_naut.get("BUY_REQUIRE_IMB", False)
    if buy_require_raw is None:
        buy_require_imb = False
    elif isinstance(buy_require_raw, str):
        buy_require_imb = buy_require_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        buy_require_imb = bool(buy_require_raw)

    buy_vol_min_raw = settings_naut.get("BUY_VOL_MIN", 1200)
    try:
        buy_vol_min = max(0, int(buy_vol_min_raw))
    except (TypeError, ValueError):
        buy_vol_min = 1200

    buy_spread_raw = settings_naut.get("BUY_SPREAD_MAX", 5.0)
    try:
        buy_spread_max = float(buy_spread_raw)
    except (TypeError, ValueError):
        buy_spread_max = 5.0

    summary_limit_raw = settings_naut.get("summary_limit", 400)
    try:
        summary_limit = max(1, int(summary_limit_raw))
    except (TypeError, ValueError):
        summary_limit = 400

    debug_raw = settings_naut.get("debug", False)
    if debug_raw is None:
        debug_enabled = False
    elif isinstance(debug_raw, str):
        debug_enabled = debug_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        debug_enabled = bool(debug_raw)

    read_window_raw = settings_naut.get("read_window_sec", 180)
    try:
        read_window_sec = max(1, int(read_window_raw))
    except (TypeError, ValueError):
        read_window_sec = 180

    db_path: str = conf.get("db_path", "rss_snapshot.db")
    symbols: List[str] = conf.get("symbols", [])
    market_window: Optional[str] = conf.get("market_window")
    vol_min_by_code: Dict[str, int] = conf.get("vol_min_by_code", {})

    profiles = {"prod": PROD, "prodlite": PRODLITE, "demo": DEMO}
    cfg = profiles[args.profile]

    log_path = "logs/naut_runner.log"
    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.info("naut_runner start (profile=%s, window=%s)", args.profile, market_window)
    logger.info(f"[RULE] SELL require_imb={sell_require_imb} thr={sell_uptick_thr}")
    logger.info(f"[RULE] BUY require_imb={buy_require_imb} thr={buy_uptick_thr} vol_min={buy_vol_min} spread_max={buy_spread_max}")


    now_dt = datetime.now()
    start_of_day = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = now_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    window_fmt = "%Y-%m-%dT%H:%M:%S"
    since = start_of_day.strftime(window_fmt)
    until = end_of_day.strftime(window_fmt)
    logger.info(f"[RULE] read_window since={since} until={until}")

    poll = FeaturePoller(db_path)
    db_abs_path: str
    try:
        pragma_rows = poll.conn.execute("PRAGMA database_list").fetchall()
        db_abs_path = next((row[2] for row in pragma_rows if len(row) >= 3 and row[1] == 'main'), None)
        if not db_abs_path and pragma_rows:
            db_abs_path = pragma_rows[0][2]
    except Exception:
        db_abs_path = str(Path(db_path).resolve())
    else:
        if not db_abs_path:
            db_abs_path = str(Path(db_path).resolve())
    COOLDOWN_SEC = 5
    last_fire_ts: Dict[str, float] = {s: 0.0 for s in symbols}
    accepted_total = 0
    accepted_per: Dict[str, Dict[str, int]] = {}
    summary_query_info_printed = False
    skips = {'cooldown': 0, 'streak': 0, 'base': 0, 'other': 0}
    skips_detail = {'vol': 0, 'spread': 0, 'uptick': 0, 'imb': 0}


    try:
        while True:
            if not print_summary and not within_window(market_window):
                time.sleep(0.2)
                continue

            for s in symbols:
                limit_for_fetch = summary_limit if print_summary else RECENT_LIMIT
                if print_summary and not summary_query_info_printed:
                    watch_preview = symbols[:3]
                    print('DBG summary db_path:', db_abs_path)
                    print('DBG summary watchlist:', len(symbols), watch_preview)
                    print('DBG summary window:', since, until)
                    query_sql = "SELECT * FROM features_stream WHERE ticker=? AND ts >= ? AND ts <= ? ORDER BY ts DESC LIMIT ?"
                    query_params = (s, since, until, limit_for_fetch)
                    print('DBG query:', query_sql, 'params:', query_params)
                    summary_query_info_printed = True
                print("DBG start query")
                rows = poll.fetch_recent(s, since, until, limit=limit_for_fetch)
                total_candidates = len(rows)
                per_candidates = {}
                for row in rows:
                    ticker = row.get('ticker')
                    if ticker is None:
                        continue
                    per_candidates[ticker] = per_candidates.get(ticker, 0) + 1
                print('DBG candidates:', total_candidates, 'per:', per_candidates)
                if not rows:
                    poll.record_direction(s, 0, cfg["WINDOW"])
                    continue

                for f in rows:
                    up_raw = f.get("uptick_ratio")
                    try:
                        up_val = float(up_raw)
                    except (TypeError, ValueError):
                        up_val = None

                    buy_gate = (up_val is not None) and (up_val >= buy_uptick_thr)
                    sell_gate = (up_val is not None) and (up_val <= (1.0 - sell_uptick_thr))

                    if not buy_gate and not sell_gate:
                        skips_detail['uptick'] += 1
                        skips['base'] += 1
                        poll.record_direction(s, 0, cfg["WINDOW"])
                        continue

                    if buy_gate and sell_gate:
                        buy_strength = up_val - buy_uptick_thr
                        sell_strength = (1.0 - sell_uptick_thr) - up_val
                        if sell_strength > buy_strength:
                            side_choice = "SELL"
                        elif buy_strength > sell_strength:
                            side_choice = "BUY"
                        else:
                            side_choice = "SELL"
                        if debug_enabled:
                            print(f"DBG overlap up={up_val:.2f} buy_s={buy_strength:.3f} sell_s={sell_strength:.3f} -> {side_choice}")
                    elif buy_gate:
                        side_choice = "BUY"
                    else:
                        side_choice = "SELL"

                    summary_detail = skips_detail if print_summary else None
                    buyish = False
                    sellish = False

                    if side_choice == "BUY":
                        buyish = is_buy_signal(
                            f,
                            vol_min_by_code,
                            buy_uptick_thr=buy_uptick_thr,
                            buy_require_imb=buy_require_imb,
                            buy_vol_min=buy_vol_min,
                            buy_spread_max=buy_spread_max,
                            skips=skips,
                            skips_detail=summary_detail,
                        )
                    else:
                        if debug_enabled:
                            spr = f.get("spread_bp")
                            vol = f.get("vol_sum")
                            up_print = up_val if up_val is not None else 0.0
                            print(f"DBG sell_check up={up_print:.2f} thr={sell_uptick_thr:.2f} gate={sell_gate} spr={spr} vol={vol}")
                        sellish = is_sell_signal(
                            f,
                            cfg,
                            vol_min_by_code,
                            sell_uptick_thr=sell_uptick_thr,
                            sell_require_imb=sell_require_imb,
                            skips=skips,
                            skips_detail=summary_detail,
                        )

                    if buyish:
                        poll.record_direction(s, +1, cfg["WINDOW"])
                        if poll.consistent(s, +1, cfg["WINDOW"]):
                            base_score = compute_base_score("BUY", f, cfg, vol_min_by_code)
                            r = apply_burst_and_decide(poll.conn, s, base_score, threshold=0.60, since=since, until=until)
                            log_decision(r)
                            if not r["decision"]:
                                if print_summary:
                                    skips['base'] += 1
                                continue
                            now = time.time()
                            if now - last_fire_ts[s] < COOLDOWN_SEC:
                                skips['cooldown'] += 1
                                continue
                            last_fire_ts[s] = now
                            log_side("BUY", f, r)
                            ticker_key = f.get('ticker') or 'UNKNOWN'
                            slot = accepted_per.setdefault(ticker_key, {'BUY': 0, 'SELL': 0})
                            slot['BUY'] += 1
                            accepted_total += 1
                            print('DBG accepted:', accepted_total, 'per:', accepted_per)
                            break
                        else:
                            skips['streak'] += 1
                            continue

                    elif sellish:
                        poll.record_direction(s, -1, cfg["WINDOW"])
                        if poll.consistent(s, -1, cfg["WINDOW"]):
                            base_score = compute_base_score("SELL", f, cfg, vol_min_by_code)
                            r = apply_burst_and_decide(poll.conn, s, base_score, threshold=0.60, since=since, until=until)
                            log_decision(r)
                            if not r["decision"]:
                                if print_summary:
                                    skips['base'] += 1
                                continue
                            now = time.time()
                            if now - last_fire_ts[s] < COOLDOWN_SEC:
                                skips['cooldown'] += 1
                                continue
                            last_fire_ts[s] = now
                            log_side("SELL", f, r)
                            ticker_key = f.get('ticker') or 'UNKNOWN'
                            slot = accepted_per.setdefault(ticker_key, {'BUY': 0, 'SELL': 0})
                            slot['SELL'] += 1
                            accepted_total += 1
                            print('DBG accepted:', accepted_total, 'per:', accepted_per)
                            break
                        else:
                            skips['streak'] += 1
                            continue

                    else:
                        poll.record_direction(s, 0, cfg["WINDOW"])
                        if print_summary:
                            skips['other'] += 1
            if print_summary:
                                skips['base'] += 1
                            continue
                        now = time.time()
                        if now - last_fire_ts[s] < COOLDOWN_SEC:
                            skips['cooldown'] += 1
                            continue
                        last_fire_ts[s] = now
                        log_side("BUY", f, r)
                        ticker_key = f.get('ticker') or 'UNKNOWN'
                        slot = accepted_per.setdefault(ticker_key, {'BUY': 0, 'SELL': 0})
                        slot['BUY'] += 1
                        accepted_total += 1
                        print('DBG accepted:', accepted_total, 'per:', accepted_per)
                        break
                    else:
                        skips['streak'] += 1
                        continue

                elif sellish:
                    poll.record_direction(s, -1, cfg["WINDOW"])
                    if poll.consistent(s, -1, cfg["WINDOW"]):
                        base_score = compute_base_score("SELL", f, cfg, vol_min_by_code)
                        r = apply_burst_and_decide(poll.conn, s, base_score, threshold=0.60, since=since, until=until)
                        log_decision(r)
                        if not r["decision"]:
                            if print_summary:
                                skips['base'] += 1
                            continue
                        now = time.time()
                        if now - last_fire_ts[s] < COOLDOWN_SEC:
                            skips['cooldown'] += 1
                            continue
                        last_fire_ts[s] = now
                        log_side("SELL", f, r)
                        ticker_key = f.get('ticker') or 'UNKNOWN'
                        slot = accepted_per.setdefault(ticker_key, {'BUY': 0, 'SELL': 0})
                        slot['SELL'] += 1
                        accepted_total += 1
                        print('DBG accepted:', accepted_total, 'per:', accepted_per)
                        break
                    else:
                        skips['streak'] += 1
                        continue

                else:
                    poll.record_direction(s, 0, cfg["WINDOW"])
                    if print_summary:
                        skips['other'] += 1
            if print_summary:
                    skips['base'] += 1
                continue
            now = time.time()
            if now - last_fire_ts[s] < COOLDOWN_SEC:
                skips['cooldown'] += 1
                continue
            last_fire_ts[s] = now
            log_side("BUY", f, r)
            ticker_key = f.get('ticker') or 'UNKNOWN'
            slot = accepted_per.setdefault(ticker_key, {'BUY': 0, 'SELL': 0})
            slot['BUY'] += 1
            accepted_total += 1
            print('DBG accepted:', accepted_total, 'per:', accepted_per)
            break
        else:
            skips['streak'] += 1
            continue

    elif sellish:
        poll.record_direction(s, -1, cfg["WINDOW"])
        if poll.consistent(s, -1, cfg["WINDOW"]):
            base_score = compute_base_score("SELL", f, cfg, vol_min_by_code)
            r = apply_burst_and_decide(poll.conn, s, base_score, threshold=0.60, since=since, until=until)
            log_decision(r)
            if not r["decision"]:
                if print_summary:
                    skips['base'] += 1
                continue
            now = time.time()
            if now - last_fire_ts[s] < COOLDOWN_SEC:
                skips['cooldown'] += 1
                continue
            last_fire_ts[s] = now
            log_side("SELL", f, r)
            ticker_key = f.get('ticker') or 'UNKNOWN'
            slot = accepted_per.setdefault(ticker_key, {'BUY': 0, 'SELL': 0})
            slot['SELL'] += 1
            accepted_total += 1
            print('DBG accepted:', accepted_total, 'per:', accepted_per)
            break
        else:
            skips['streak'] += 1
            continue

    else:
        poll.record_direction(s, 0, cfg["WINDOW"])
        if print_summary:
            skips['other'] += 1
            if print_summary:
                                    skips['base'] += 1
                                continue
                            now = time.time()
                            if now - last_fire_ts[s] < COOLDOWN_SEC:
                                skips['cooldown'] += 1
                                continue
                            last_fire_ts[s] = now
                            log_side("BUY", f, r)
                            ticker_key = f.get('ticker') or 'UNKNOWN'
                            slot = accepted_per.setdefault(ticker_key, {'BUY': 0, 'SELL': 0})
                            slot['BUY'] += 1
                            accepted_total += 1
                            print('DBG accepted:', accepted_total, 'per:', accepted_per)
                            break
                        else:
                            skips['streak'] += 1
                            continue

                    elif sellish:
                        poll.record_direction(s, -1, cfg["WINDOW"])
                        if poll.consistent(s, -1, cfg["WINDOW"]):
                            base_score = compute_base_score("SELL", f, cfg, vol_min_by_code)
                            r = apply_burst_and_decide(poll.conn, s, base_score, threshold=0.60, since=since, until=until)
                            log_decision(r)
                            if not r["decision"]:
                                if print_summary:
                                    skips['base'] += 1
                                continue
                            now = time.time()
                            if now - last_fire_ts[s] < COOLDOWN_SEC:
                                skips['cooldown'] += 1
                                continue
                            last_fire_ts[s] = now
                            log_side("SELL", f, r)
                            ticker_key = f.get('ticker') or 'UNKNOWN'
                            slot = accepted_per.setdefault(ticker_key, {'BUY': 0, 'SELL': 0})
                            slot['SELL'] += 1
                            accepted_total += 1
                            print('DBG accepted:', accepted_total, 'per:', accepted_per)
                            break
                        else:
                            skips['streak'] += 1
                            continue

                    else:
                        poll.record_direction(s, 0, cfg["WINDOW"])
                        if print_summary:
                            skips['other'] += 1

            if print_summary:
                break
            time.sleep(0.25)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
    finally:
        if print_summary:
            print('DBG skipped:', skips)
            print('DBG skipped_detail:', skips_detail)
            print('DBG accepted:', accepted_total, 'per:', accepted_per)
        logger.info("naut_runner stop")

if __name__ == "__main__":
    main()




