#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_board_path.py
  - kabuステーション REST /board で板情報を取得
  - best bid/ask と top3 数量を正規化
  - --write 指定で orderbook_snapshot に UPSERT 保存

使い方（PowerShell）:
  py .\diag_board_path.py -Config config\stream_settings.json --loops 1
  py .\diag_board_path.py -Config config\stream_settings.json --loops 3 --sleep 1 --write
"""

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime
from urllib import request

# ---------- utils ----------

def load_json_utf8(path: str) -> dict:
    # UTF-8 / UTF-8 with BOM の両方に対応
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

# ---------- HTTP ----------

def fetch_board(host: str, port: int, token: str, symbol: str) -> dict:
    """
    REST /board を取得（@1＝現値気配）。addinfo=true を付けると配列/段の両方が揃う環境が多いです。
    """
    url = f"http://{host}:{port}/kabusapi/board/{symbol}@1?addinfo=true"
    req = request.Request(url, headers={"X-API-KEY": token})
    try:
        with request.urlopen(req, timeout=5) as res:
            return json.loads(res.read().decode("utf-8"))
    except Exception as e:
        return {"HTTP_ERR": str(e)}

# ---------- normalize ----------

def _collect_bids_asks(data: dict):
    """
    返却JSONのバリエーションを吸収して bids/asks を [(price, qty), ...] で返す。
    対応パターン：
      - 配列: Bids/Asks, buys/sells（要素は {Price, Qty} など）
      - 段   : Buy1..Buy10 / Sell1..Sell10（要素は {Price, Qty} など）
    """
    bids, asks = [], []

    # 配列（大文字）
    if isinstance(data.get("Bids"), list):
        for r in data["Bids"]:
            if isinstance(r, dict):
                p = _to_float(r.get("Price"))
                q = _to_int(r.get("Qty") or r.get("Quantity"))
                if p is not None:
                    bids.append((p, q))
    if isinstance(data.get("Asks"), list):
        for r in data["Asks"]:
            if isinstance(r, dict):
                p = _to_float(r.get("Price"))
                q = _to_int(r.get("Qty") or r.get("Quantity"))
                if p is not None:
                    asks.append((p, q))

    # 配列（小文字）
    if not bids and isinstance(data.get("buys"), list):
        for r in data["buys"]:
            if isinstance(r, dict):
                p = _to_float(r.get("Price") or r.get("price"))
                q = _to_int(r.get("Qty") or r.get("Quantity") or r.get("qty"))
                if p is not None:
                    bids.append((p, q))
    if not asks and isinstance(data.get("sells"), list):
        for r in data["sells"]:
            if isinstance(r, dict):
                p = _to_float(r.get("Price") or r.get("price"))
                q = _to_int(r.get("Qty") or r.get("Quantity") or r.get("qty"))
                if p is not None:
                    asks.append((p, q))

    # 段（Buy1.. / Sell1..）
    if not bids:
        for i in range(1, 11):
            k = f"Buy{i}"
            r = data.get(k)
            if isinstance(r, dict):
                p = _to_float(r.get("Price") or r.get("price"))
                q = _to_int(r.get("Qty") or r.get("Quantity") or r.get("qty"))
                if p is not None:
                    bids.append((p, q))
    if not asks:
        for i in range(1, 11):
            k = f"Sell{i}"
            r = data.get(k)
            if isinstance(r, dict):
                p = _to_float(r.get("Price") or r.get("price"))
                q = _to_int(r.get("Qty") or r.get("Quantity") or r.get("qty"))
                if p is not None:
                    asks.append((p, q))

    return bids, asks

def parse_board(symbol: str, board: dict):
    """板レスポンスから保存対象を抽出（堅牢版）"""
    if "HTTP_ERR" in board:
        return None, [f"http_err:{board['HTTP_ERR']}"]

    reasons = []

    bids, asks = _collect_bids_asks(board)

    bid1 = max([p for p, _ in bids]) if bids else None
    ask1 = min([p for p, _ in asks]) if asks else None

    # フォールバック（単発キー）
    if bid1 is None:
        bid1 = _to_float(board.get("BestBidPrice") or board.get("BidPrice"))
    if ask1 is None:
        ask1 = _to_float(board.get("BestAskPrice") or board.get("AskPrice"))

    if bid1 is None:
        reasons.append("no_bid1")
    if ask1 is None:
        reasons.append("no_ask1")
    if reasons:
        return None, reasons

    if ask1 < bid1:
        return None, ["ask_lt_bid"]

    # top3 数量
    buy_top3 = sum(q for _, q in sorted(bids, reverse=True)[:3]) if bids else 0
    sell_top3 = sum(q for _, q in sorted(asks, reverse=False)[:3]) if asks else 0

    # スプレッド（bp）
    mid = (bid1 + ask1) / 2.0
    spread_bp = ((ask1 - bid1) / mid * 10000.0) if (mid and mid > 0) else None

    return {
        "ticker": symbol,
        "bid1": bid1,
        "ask1": ask1,
        "spread_bp": spread_bp,
        "buy_top3": buy_top3,
        "sell_top3": sell_top3,
    }, []

# ---------- DB ----------

def ensure_tables(cur: sqlite3.Cursor):
    cur.execute("""
    create table if not exists orderbook_snapshot(
      ticker    text not null,
      ts        text not null,
      bid1      real,
      ask1      real,
      spread_bp real,
      buy_top3  integer,
      sell_top3 integer,
      primary key (ticker, ts)
    )
    """)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True, help="settings JSON")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cfg = load_json_utf8(args.Config)
    host = cfg.get("host", "localhost")
    port = int(cfg.get("port", 18080))
    symbols = cfg.get("symbols") or []
    if not symbols:
        raise SystemExit("symbols is empty in config")
    # トークンは環境変数優先（kabus_login_wait.py 実行済み想定）
    token = os.environ.get("KABU_TOKEN") or cfg.get("token")
    if not token:
        raise SystemExit("KABU_TOKEN not found. run kabus_login_wait.py or set token in config.")

    con = None
    cur = None
    if args.write:
        db_path = cfg.get("db_path", "rss_snapshot.db")
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        ensure_tables(cur)
        con.commit()

    for i in range(max(1, args.loops)):
        for sym in symbols:
            board = fetch_board(host, port, token, sym)
            row, reasons = parse_board(sym, board)
            if row is None:
                # 取得できなかった理由と、応答に含まれる主なキーを出す
                keys = list(board.keys())
                print(f"[{sym}] FAIL reasons={reasons} keys={keys}")
                continue

            print(f"[{sym}] OK bid1={row['bid1']} ask1={row['ask1']} spbp={row['spread_bp']:.2f} top3=({row['buy_top3']},{row['sell_top3']})")

            if args.write and cur is not None:
                now_iso = datetime.now().isoformat(timespec="seconds")
                cur.execute("""
                    insert or replace into orderbook_snapshot
                      (ticker, ts, bid1, ask1, spread_bp, buy_top3, sell_top3)
                    values (?,?,?,?,?,?,?)
                """, (sym, now_iso, row["bid1"], row["ask1"], row["spread_bp"], row["buy_top3"], row["sell_top3"]))
                con.commit()

        if args.loops > 1 and i < args.loops - 1:
            time.sleep(args.sleep)

    if con is not None:
        con.close()

if __name__ == "__main__":
    main()
