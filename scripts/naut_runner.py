# scripts/naut_runner.py

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

from scripts.burst_helper import burst_bonus


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
    logging.info(
        f"[{side}] {f['ticker']} ts={f['ts']} "
        f"uptick={f['uptick_ratio']:.2f} imb={f['depth_imbalance']:.2f} "
        f"spr={spr_txt} vol={f['vol_sum']:.0f} | "
        f"base={r['base']:.3f} adj={r['adjusted']:.3f} bonus={r['bonus']:.3f} thr={r['threshold']:.2f}"
    )


def apply_burst_and_decide(conn, ticker: str, base_score: float, threshold: float = 0.60):
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
           WHERE ticker=? AND (burst_buy=1 OR burst_sell=1)
           ORDER BY ts DESC LIMIT 1""",
        (ticker,),
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
    logging.debug(
        "burst_decision | %(ticker)s | base=%(base).3f -> adj=%(adjusted).3f (+%(bonus).3f) "
        "thr=%(threshold).2f cooled=%(cooled)s decision=%(decision)s ts=%(burst_ts)s",
        r,
    )


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

    def fetch_recent(self, symbol: str) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT * FROM features_stream WHERE ticker=? ORDER BY ts DESC LIMIT ?",
            (symbol, RECENT_LIMIT),
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


def is_buy_signal(f: Dict, cfg: Dict, vol_min_by_code: Dict[str, int]) -> bool:
    vol_min = vol_min_by_code.get(f["ticker"], cfg["VOL_MIN"])
    if f["vol_sum"] < vol_min:
        return False
    if not spread_ok(f["spread_bp"], cfg["BUY"]["spread"], cfg["ALLOW_SPREAD_NONE"]):
        return False
    if f["uptick_ratio"] < cfg["BUY"]["uptick"]:
        return False
    if f["depth_imbalance"] < cfg["BUY"]["imb"]:
        return False
    return True


def is_sell_signal(f: Dict, cfg: Dict, vol_min_by_code: Dict[str, int]) -> bool:
    vol_min = vol_min_by_code.get(f["ticker"], cfg["VOL_MIN"])
    if f["vol_sum"] < vol_min:
        return False
    if not spread_ok(f["spread_bp"], cfg["SELL"]["spread"], cfg["ALLOW_SPREAD_NONE"]):
        return False
    if f["uptick_ratio"] > cfg["SELL"]["uptick"]:
        return False
    if f["depth_imbalance"] > cfg["SELL"]["imb"]:
        return False
    return True

# ------------------------------- main loop ------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    ap.add_argument("--profile", choices=["prod", "prodlite", "demo"], default="prod")
    args = ap.parse_args()

    with open(args.Config, "r", encoding="utf-8-sig") as f:
        conf = json.load(f)

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
    logging.info("naut_runner start (profile=%s, window=%s)", args.profile, market_window)

    poll = FeaturePoller(db_path)
    COOLDOWN_SEC = 5
    last_fire_ts: Dict[str, float] = {s: 0.0 for s in symbols}

    try:
        while True:
            if not within_window(market_window):
                time.sleep(0.2)
                continue

            for s in symbols:
                rows = poll.fetch_recent(s)
                if not rows:
                    poll.record_direction(s, 0, cfg["WINDOW"])
                    continue

                for f in rows:
                    buyish = is_buy_signal(f, cfg, vol_min_by_code)
                    sellish = is_sell_signal(f, cfg, vol_min_by_code)

                    if buyish:
                        poll.record_direction(s, +1, cfg["WINDOW"])
                        if poll.consistent(s, +1, cfg["WINDOW"]):
                            base_score = compute_base_score("BUY", f, cfg, vol_min_by_code)
                            r = apply_burst_and_decide(poll.conn, s, base_score, threshold=0.60)
                            log_decision(r)
                            if not r["decision"]:
                                continue
                            now = time.time()
                            if now - last_fire_ts[s] < COOLDOWN_SEC:
                                continue
                            last_fire_ts[s] = now
                            log_side("BUY", f, r)
                            break

                    elif sellish:
                        poll.record_direction(s, -1, cfg["WINDOW"])
                        if poll.consistent(s, -1, cfg["WINDOW"]):
                            base_score = compute_base_score("SELL", f, cfg, vol_min_by_code)
                            r = apply_burst_and_decide(poll.conn, s, base_score, threshold=0.60)
                            log_decision(r)
                            if not r["decision"]:
                                continue
                            now = time.time()
                            if now - last_fire_ts[s] < COOLDOWN_SEC:
                                continue
                            last_fire_ts[s] = now
                            log_side("SELL", f, r)
                            break

                    else:
                        poll.record_direction(s, 0, cfg["WINDOW"])

            time.sleep(0.25)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    finally:
        logging.info("naut_runner stop")


if __name__ == "__main__":
    main()
