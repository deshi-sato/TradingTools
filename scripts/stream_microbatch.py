import argparse
import json
import logging
import queue
import sqlite3
import sys
import threading
import time
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Tuple, Optional

from feature_calc import top3_sum, spread_bp, depth_imbalance, uptick_ratio
from board_fetcher import BoardFetcher

# =========================
# DB helpers
# =========================
DDL_TICK_BATCH = """
CREATE TABLE IF NOT EXISTS tick_batch (
  ticker TEXT,
  ts_window_start TEXT,
  ts_window_end   TEXT,
  ticks           INT,
  upticks         INT,
  downticks       INT,
  vol_sum         REAL,
  last_price      REAL
);
"""
DDL_OB_SNAP = """
CREATE TABLE IF NOT EXISTS orderbook_snapshot (
  ticker TEXT,
  ts     TEXT,
  bid1   REAL,
  ask1   REAL,
  spread_bp REAL,
  buy_top3  INT,
  sell_top3 INT
);
"""
DDL_FEAT = """
CREATE TABLE IF NOT EXISTS features_stream (
  ticker TEXT,
  ts     TEXT,
  uptick_ratio REAL,
  vol_sum      REAL,
  spread_bp    REAL,
  buy_top3     INT,
  sell_top3    INT,
  depth_imbalance REAL
);
"""


def ensure_tables(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    with conn:
        conn.executescript(DDL_TICK_BATCH + DDL_OB_SNAP + DDL_FEAT)
    conn.close()


def insert_tick_batch(conn: sqlite3.Connection, rows: List[Tuple]) -> None:
    conn.executemany("INSERT INTO tick_batch VALUES (?,?,?,?,?,?,?,?)", rows)


def insert_orderbook(conn: sqlite3.Connection, rows: List[Tuple]) -> None:
    conn.executemany("INSERT INTO orderbook_snapshot VALUES (?,?,?,?,?,?,?)", rows)


def insert_features(conn: sqlite3.Connection, rows: List[Tuple]) -> None:
    conn.executemany("INSERT INTO features_stream VALUES (?,?,?,?,?,?,?,?)", rows)


# =========================
# Tick receiver (ダミー)
# =========================
class TickReceiver(threading.Thread):
    """
    kabuステ PUSH に置き換える想定。
    最小構成ではダミーで価格を小刻みに上下させる。
    queue へ (symbol, price, qty, ts_iso) をput。
    """

    def __init__(
        self,
        symbols: List[str],
        q: "queue.Queue",
        stop_event: threading.Event,
        interval_ms: int = 50,
    ):
        super().__init__(daemon=True)
        self.symbols = symbols
        self.q = q
        self.stop_event = stop_event
        self.interval = interval_ms / 1000.0
        self._last_price: Dict[str, float] = {s: 1000.0 for s in symbols}

    def run(self) -> None:
        import random

        while not self.stop_event.is_set():
            for s in self.symbols:
                base = self._last_price[s]
                delta = random.choice([-0.1, 0.0, 0.1])
                price = round(base + delta, 1)
                qty = random.randint(1, 5) * 100
                ts_iso = datetime.now().isoformat(timespec="milliseconds")
                self.q.put((s, price, qty, ts_iso))
                self._last_price[s] = price
            time.sleep(self.interval)


# =========================
# Market window helper
# =========================
def within_market_window(spec: Optional[str]) -> bool:
    if not spec:
        return True
    try:
        start_s, end_s = spec.split("-")

        def to_t(s: str) -> dtime:
            hh, mm = s.split(":")
            return dtime(int(hh), int(mm))

        now = datetime.now().time()
        return to_t(start_s) <= now <= to_t(end_s)
    except Exception:
        return True


# =========================
# Main worker
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Verbose", type=int, default=1)
    args = ap.parse_args()

    with open(args.Config, "r", encoding="utf-8-sig") as f:
        conf = json.load(f)
    window_ms: int = conf.get("window_ms", 300)
    symbols: List[str] = conf.get("symbols", [])
    db_path: str = conf.get("db_path", "rss_snapshot.db")
    log_path: str = conf.get("log_path", "logs/stream_microbatch.log")
    board_mode: str = conf.get("board_mode", "auto")
    rest_poll_ms: int = conf.get("rest_poll_ms", 500)
    market_window: Optional[str] = conf.get("market_window")

    logging.basicConfig(
        level=logging.INFO if args.Verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("🚀 stream_microbatch start")
    logging.info(
        f"config: window_ms={window_ms} symbols={symbols} db={db_path} board_mode={board_mode}"
    )

    ensure_tables(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    q: "queue.Queue" = queue.Queue(maxsize=conf.get("tick_queue_max", 20000))
    stop_event = threading.Event()

    # Tick receiver 起動（後でPUSHに置換）
    tick_thread = TickReceiver(symbols, q, stop_event, interval_ms=50)
    tick_thread.start()

    board = BoardFetcher(mode=board_mode, rest_poll_ms=rest_poll_ms)

    # 集計用状態
    last_price: Dict[str, Optional[float]] = {s: None for s in symbols}

    window_s = window_ms / 1000.0
    next_cut = time.monotonic() + window_s

    try:
        while True:
            if market_window and not within_market_window(market_window):
                # 市場時間外は少し待機（10:00で停止イメージ）
                time.sleep(0.2)
                if datetime.now().time() > dtime(10, 0):
                    logging.info("⏹ 市場時間終了: flushing & exit")
                    break

            # バッチ用バッファ
            ticks_buf: Dict[str, List[Tuple[float, int, str]]] = {
                s: [] for s in symbols
            }

            # ウィンドウまでドレイン
            while time.monotonic() < next_cut:
                try:
                    s, price, qty, ts_iso = q.get(timeout=0.01)
                    if s in ticks_buf:
                        ticks_buf[s].append((price, qty, ts_iso))
                except queue.Empty:
                    pass

            # ウィンドウ締め
            ts_start_iso = datetime.now().isoformat(timespec="milliseconds")
            tick_rows = []
            ob_rows = []
            feat_rows = []

            for s in symbols:
                arr = ticks_buf[s]
                if not arr:
                    # 板だけでもスナップショットしておく
                    ob = board.get_board(s)
                    b1, a1 = ob.get("bid1"), ob.get("ask1")
                    spr = spread_bp(b1, a1)
                    buy3 = top3_sum(ob.get("bids"))
                    sell3 = top3_sum(ob.get("asks"))
                    ob_rows.append((s, ts_start_iso, b1, a1, spr, buy3, sell3))
                    continue

                # upticks/downticks/vol_sum/last_price
                upt = dwn = 0
                vol_sum = 0.0
                prev = last_price[s] if last_price[s] is not None else arr[0][0]
                for price, qty, _ in arr:
                    if price > prev:
                        upt += 1
                    elif price < prev:
                        dwn += 1
                    prev = price
                    vol_sum += qty
                last = arr[-1][0]
                last_price[s] = last

                ts_end_iso = arr[-1][2]  # 最後のティック時刻（ISO文字列想定）
                tick_rows.append(
                    (s, ts_start_iso, ts_end_iso, len(arr), upt, dwn, vol_sum, last)
                )

                # 板
                ob = board.get_board(s)
                b1, a1 = ob.get("bid1"), ob.get("ask1")
                spr = spread_bp(b1, a1)
                buy3 = top3_sum(ob.get("bids"))
                sell3 = top3_sum(ob.get("asks"))
                ob_rows.append((s, ts_end_iso, b1, a1, spr, buy3, sell3))

                # 特徴量
                feat_rows.append(
                    (
                        s,
                        ts_end_iso,
                        uptick_ratio(upt, dwn),
                        vol_sum,
                        spr,
                        buy3,
                        sell3,
                        depth_imbalance(buy3, sell3),
                    )
                )

            # DB書き込み
            with conn:
                if tick_rows:
                    insert_tick_batch(conn, tick_rows)
                if ob_rows:
                    insert_orderbook(conn, ob_rows)
                if feat_rows:
                    insert_features(conn, feat_rows)

            # 監視用ログ（軽め）
            total_ticks = sum(r[3] for r in tick_rows) if tick_rows else 0
            logging.info(
                f"batch ticks={total_ticks} ob_snaps={len(ob_rows)} feats={len(feat_rows)}"
            )

            # 次ウィンドウへ
            now_mono = time.monotonic()
            next_cut += window_s
            if next_cut < now_mono:
                # 遅延が蓄積していたら追いつく
                next_cut = now_mono + window_s

    except KeyboardInterrupt:
        logging.info("⛔ KeyboardInterrupt")
    finally:
        stop_event.set()
        tick_thread.join(timeout=1.0)
        conn.close()
        logging.info("✅ stream_microbatch stop")


if __name__ == "__main__":
    main()
