# scripts/naut_runner.py  （完全版・上書き用）

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

# ------------ ユーティリティ ------------

def within_window(spec: Optional[str]) -> bool:
    """設定の 'HH:MM-HH:MM' を見て現在時刻が窓内か判定。None/空なら常時True。"""
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

# ------------ プロファイル（閾値） ------------

# 本番想定（板・厚みも使う。spread None は不許可）
PROD = dict(
    BUY=dict(uptick=0.62, imb=0.15, spread=3.0),
    SELL=dict(uptick=0.38, imb=-0.15, spread=3.0),
    VOL_MIN=600,
    WINDOW=5,
    ALLOW_SPREAD_NONE=False,
)

# 検証用ゆるめ（確認専用）
DEMO = dict(
    BUY=dict(uptick=0.40, imb=-1.0, spread=9999.0),
    SELL=dict(uptick=0.60, imb= 1.0, spread=9999.0),
    VOL_MIN=0,
    WINDOW=1,
    ALLOW_SPREAD_NONE=True,
)

# 本番寄り：出すぎ抑制のため少し厳しめ＋板欠落は原則NG
PRODLITE = dict(
    BUY=dict(uptick=0.65, imb=0.00, spread=5.0),
    SELL=dict(uptick=0.35, imb=-0.00, spread=5.0),
    VOL_MIN=1200,
    WINDOW=4,
    ALLOW_SPREAD_NONE=False,  # spread=None は通さない
)

# 直近N行を審査：どれかが条件を満たしたら方向バッファに+1/-1を積む
RECENT_LIMIT = 5
MAX_SLIPPAGE_BP = 4.0  # いまは未使用（将来の発注前チェック用）

# ------------ DB ポーラ ------------

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
        # 多数決（半分超え）
        return agree >= (window_count // 2 + 1)

# ------------ 判定ロジック ------------

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

# ------------ メイン ------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    ap.add_argument("--profile", choices=["prod", "prodlite", "demo"], default="prod")
    args = ap.parse_args()

    # 設定読込（BOMありでもOK）
    with open(args.Config, "r", encoding="utf-8-sig") as f:
        conf = json.load(f)

    db_path: str = conf.get("db_path", "rss_snapshot.db")
    symbols: List[str] = conf.get("symbols", [])
    market_window: Optional[str] = conf.get("market_window")
    vol_min_by_code: Dict[str, int] = conf.get("vol_min_by_code", {})

    # プロファイル選択
    profiles = {"prod": PROD, "prodlite": PRODLITE, "demo": DEMO}
    cfg = profiles[args.profile]

    # ログ設定（絵文字なし／UTF-8）
    log_path = "logs/naut_runner.log"
    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("naut_runner start (profile=%s, window=%s)", args.profile, market_window)

    poll = FeaturePoller(db_path)

    # ★ 連発抑制：銘柄ごとクールダウン（秒）
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

                # 直近行を走査して buy/sell どちらかが成立したら方向バッファに積む
                for f in rows:
                    buyish = is_buy_signal(f, cfg, vol_min_by_code)
                    sellish = is_sell_signal(f, cfg, vol_min_by_code)

                    if buyish:
                        poll.record_direction(s, +1, cfg["WINDOW"])
                        if poll.consistent(s, +1, cfg["WINDOW"]):
                            now = time.time()
                            if now - last_fire_ts[s] < COOLDOWN_SEC:
                                continue  # 連発抑制
                            last_fire_ts[s] = now
                            logging.info(
                                "[BUY] %s ts=%s uptick=%.2f imb=%.2f spr=%s vol=%.0f",
                                s, f["ts"], f["uptick_ratio"], f["depth_imbalance"],
                                (None if f['spread_bp'] is None else f"{f['spread_bp']:.1f}"),
                                f["vol_sum"],
                            )
                            break

                    elif sellish:
                        poll.record_direction(s, -1, cfg["WINDOW"])
                        if poll.consistent(s, -1, cfg["WINDOW"]):
                            now = time.time()
                            if now - last_fire_ts[s] < COOLDOWN_SEC:
                                continue
                            last_fire_ts[s] = now
                            logging.info(
                                "[SELL] %s ts=%s uptick=%.2f imb=%.2f spr=%s vol=%.0f",
                                s, f["ts"], f["uptick_ratio"], f["depth_imbalance"],
                                (None if f['spread_bp'] is None else f"{f['spread_bp']:.1f}"),
                                f["vol_sum"],
                            )
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
