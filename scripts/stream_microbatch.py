# -*- coding: utf-8 -*-
"""
stream_microbatch.py
REST /board をポーリングし、板のスナップショットを SQLite に保存する正式版。

■ できること
- 設定ファイル（--Config）から、host/port/symbols/poll などを読み込む
- /board 応答から bid1/ask1、上位3本の出来高合計、スプレッドbp を正規化
- orderbook_snapshot(ticker, ts, bid1, ask1, spread_bp, buy_top3, sell_top3) に upsert
- price_guard（値の欠損、負値、ask<bid など）で不正データをスキップ
- --probe-board で /board 疎通チェック
- --loops/--sleep で実行回数やインターバルを制御（未指定なら無限ループ）

■ 依存
- 標準ライブラリのみ（sqlite3, json, urllib, argparse, datetime, time, logging）

■ 既存 DB
- 既存の rss_snapshot.db にテーブルがなければ自動作成
- 既存の orderbook_snapshot と互換（ticker, ts の UNIQUE に対応）
"""

from __future__ import annotations
import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Tuple, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

JST = timezone(timedelta(hours=9))

# ---------- 設定読み込み（UTF-8-BOM 対応） ----------
def load_json_utf8(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        raw = f.read()
    # BOM を取り除いて読み込む
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return json.loads(raw.decode("utf-8"))


# ---------- /board 取得 ----------
def fetch_board(host: str, port: int, token: str, symbol: str, timeout: int = 5) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = f"http://{host}:{port}/kabusapi/board/{symbol}@1"
    req = Request(url, headers={"X-API-KEY": token})
    try:
        with urlopen(req, timeout=timeout) as res:
            body = res.read().decode("utf-8")
            return json.loads(body), None
    except HTTPError as e:
        return None, f"HTTP_ERR:{e.code} {e.reason}"
    except URLError as e:
        return None, f"URL_ERR:{e.reason}"
    except Exception as e:
        return None, f"EXC:{e}"


# ---------- 正規化ユーティリティ ----------
def _get_first_pxqty_from_flat(root: Dict[str, Any], side_prefix: str) -> Optional[Tuple[float, float]]:
    """
    フラット（Sell1/Buy1 形式）応答から最良価格を抽出
    例）Sell1: {"Price": 2247, "Qty": 100} / Buy1: {"Price": 2246, "Qty": 200}
    """
    key = f"{'Sell' if side_prefix=='ask' else 'Buy'}1"
    if key in root and isinstance(root[key], dict):
        px = root[key].get("Price")
        qty = root[key].get("Qty")
        if isinstance(px, (int, float)) and isinstance(qty, (int, float)):
            return float(px), float(qty)
    return None


def _sum_top3_from_flat(root: Dict[str, Any], side_prefix: str) -> int:
    """
    フラット（Sell1..Sell10 / Buy1..Buy10）から上位3本の数量合計
    """
    total = 0
    side_name = "Sell" if side_prefix == "ask" else "Buy"
    for i in (1, 2, 3):
        k = f"{side_name}{i}"
        if k in root and isinstance(root[k], dict):
            qty = root[k].get("Qty")
            if isinstance(qty, (int, float)):
                total += int(qty)
    return total


def _get_first_pxqty_from_list(root: Dict[str, Any], side_prefix: str) -> Optional[Tuple[float, float]]:
    """
    リスト（"Asks":[{"Price":..,"Qty":..},..], "Bids":[...]）応答から最良価格を抽出
    """
    arr_key = "Asks" if side_prefix == "ask" else "Bids"
    arr = root.get(arr_key)
    if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
        px = arr[0].get("Price")
        qty = arr[0].get("Qty")
        if isinstance(px, (int, float)) and isinstance(qty, (int, float)):
            return float(px), float(qty)
    return None


def _sum_top3_from_list(root: Dict[str, Any], side_prefix: str) -> int:
    """
    リスト形式から上位3本の数量合計
    """
    total = 0
    arr_key = "Asks" if side_prefix == "ask" else "Bids"
    arr = root.get(arr_key)
    if isinstance(arr, list):
        for i in range(min(3, len(arr))):
            q = arr[i].get("Qty") if isinstance(arr[i], dict) else None
            if isinstance(q, (int, float)):
                total += int(q)
    return total


def normalize_board(root: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    /board 応答から、bid1/ask1、top3合計、spread_bp、タイムスタンプ ts を作る
    返り値が None の場合は price_guard に引っかかった想定（上位層でログしてスキップ）
    """
    # --- bid1/ask1 ---
    # flat 形式優先 → list 形式 fallback
    ask = _get_first_pxqty_from_flat(root, "ask") or _get_first_pxqty_from_list(root, "ask")
    bid = _get_first_pxqty_from_flat(root, "bid") or _get_first_pxqty_from_list(root, "bid")

    if not ask or not bid:
        return None
    ask1, _askq = ask
    bid1, _bidq = bid

    # --- price guard ---
    if not (isinstance(ask1, (int, float)) and isinstance(bid1, (int, float))):
        return None
    if ask1 <= 0 or bid1 <= 0:
        return None
    if ask1 < bid1:
        return None

    # --- 上位3本合計 ---
    sell_top3 = _sum_top3_from_flat(root, "ask")
    if sell_top3 == 0:
        sell_top3 = _sum_top3_from_list(root, "ask")

    buy_top3 = _sum_top3_from_flat(root, "bid")
    if buy_top3 == 0:
        buy_top3 = _sum_top3_from_list(root, "bid")

    # --- ts（できるだけ API の時刻、無ければ現在JST）---
    # CurrentPriceTime / AskTime / BidTime のうち妥当な方を採用
    def _pick_time(fields):
        for k in fields:
            v = root.get(k)
            if isinstance(v, str) and len(v) >= 19:
                return v
        return None

    ts_api = _pick_time(["CurrentPriceTime", "AskTime", "BidTime"])
    if ts_api:
        ts = ts_api
    else:
        ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
        # SQLite 側の見やすさ合わせ（+09:00 → +09:00 形式に整える必要がなければこのまま）
        if len(ts) == 24 and ts.endswith("+0900"):
            ts = ts[:-5] + "+09:00"

    # --- spread_bp（bp=0.01%）: (ask - bid) / ask * 10_000 ---
    try:
        spread_bp = float(ask1 - bid1) / float(ask1) * 10_000.0
    except Exception:
        return None

    return {
        "ts": ts,
        "bid1": float(bid1),
        "ask1": float(ask1),
        "spread_bp": float(spread_bp),
        "buy_top3": int(buy_top3),
        "sell_top3": int(sell_top3),
    }


# ---------- DB ----------
DDL_ORDERBOOK = """
CREATE TABLE IF NOT EXISTS orderbook_snapshot(
    ticker     TEXT NOT NULL,
    ts         TEXT NOT NULL,
    bid1       REAL,
    ask1       REAL,
    spread_bp  REAL,
    buy_top3   INTEGER,
    sell_top3  INTEGER,
    UNIQUE(ticker, ts)
)
"""

def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(DDL_ORDERBOOK)
    return con

def upsert_orderbook(con: sqlite3.Connection, ticker: str, rec: Dict[str, Any]) -> bool:
    # 既存 UNIQUE(ticker, ts) に対して INSERT OR IGNORE で衝突を回避
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO orderbook_snapshot
            (ticker, ts, bid1, ask1, spread_bp, buy_top3, sell_top3)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, rec["ts"], rec["bid1"], rec["ask1"], rec["spread_bp"], rec["buy_top3"], rec["sell_top3"]),
    )
    return cur.rowcount > 0


# ---------- 実行 ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True, help="path to stream_settings.json")
    ap.add_argument("--loops", type=int, default=0, help="回数指定（0=無限）")
    ap.add_argument("--sleep", type=float, default=0.5, help="各ループ間の sleep 秒")
    ap.add_argument("--probe-board", action="store_true", help="/board 疎通チェックのみを実施")
    args = ap.parse_args()

    cfg = load_json_utf8(args.Config)

    # 必須系
    host = cfg.get("host", "localhost")
    port = int(cfg.get("port", 18080))
    token = cfg.get("KABU_TOKEN") or cfg.get("kabu_token") or cfg.get("api_key") or ""
    symbols = list(cfg.get("symbols") or [])
    db_path = cfg.get("db_path") or cfg.get("db") or "rss_snapshot.db"
    rest_poll_ms = int(cfg.get("rest_poll_ms", 500))  # 0.5秒既定

    # ロガ
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("stream_microbatch start host=%s port=%s db=%s symbols=%s poll_ms=%s",
                 host, port, db_path, symbols, rest_poll_ms)

    # /board 疎通
    if args.probe_board:
        if not symbols:
            logging.error("symbols が空です。Config を確認してください。")
            sys.exit(2)
        data, err = fetch_board(host, port, token, symbols[0])
        if err:
            logging.error("PROBE /board %s -> %s", symbols[0], err)
            sys.exit(3)
        norm = normalize_board(data or {})
        if not norm:
            logging.error("PROBE 正規化失敗：%s", symbols[0])
            sys.exit(4)
        logging.info("PROBE board OK: %s bid=%.1f ask=%.1f spr=%.2fbp ts=%s",
                     symbols[0], norm["bid1"], norm["ask1"], norm["spread_bp"], norm["ts"])
        sys.exit(0)

    if not symbols:
        logging.error("symbols が空です。Config を確認してください。")
        sys.exit(1)

    con = open_db(db_path)

    loops = 0
    try:
        while True:
            loops += 1
            ins_count = 0
            for sym in symbols:
                data, err = fetch_board(host, port, token, sym)
                if err:
                    logging.warning("[HTTP] %s %s", sym, err)
                    continue
                norm = normalize_board(data or {})
                if not norm:
                    logging.debug("[GUARD] %s invalid board skip", sym)
                    continue
                if upsert_orderbook(con, sym, norm):
                    ins_count += 1

            if ins_count:
                con.commit()

            logging.info("batch ob_snaps=%s", ins_count)

            # 指定回数実行したら終了
            if args.loops and loops >= args.loops:
                break

            # 1回のバッチを 500ms〜 既定間隔で回す
            sleep_sec = max(args.sleep, rest_poll_ms / 1000.0) if args.sleep == 0.5 else args.sleep
            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    finally:
        try:
            con.commit()
        except Exception:
            pass
        con.close()
        logging.info("stream_microbatch stop")


if __name__ == "__main__":
    main()
