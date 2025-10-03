#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ws_print_ticker.py
WebSocket配信を監視して Tick/Board を表示
"""

import argparse
import json
import sys
from websocket import create_connection

def looks_like_tick(payload: dict) -> bool:
    # Symbol が必須
    if "Symbol" not in payload:
        return False
    # Price + 時刻
    if "Price" in payload and ("ExecutionDateTime" in payload or "CurrentPriceTime" in payload):
        return True
    # CurrentPrice + TradingVolume もティックとして扱う
    if "CurrentPrice" in payload and ("TradingVolume" in payload or "Volume" in payload):
        return True
    return False

def looks_like_board(payload: dict) -> bool:
    return any(k in payload for k in ("OverSellQty", "UnderBuyQty", "BidPrice", "AskPrice"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("--all", action="store_true", help="全メッセージを表示")
    ap.add_argument("--pretty", action="store_true", help="整形表示")
    args = ap.parse_args()

    # 設定ファイルから token / port を読む
    with open(args.Config, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    port = int(cfg.get("port", 18080))
    token = cfg.get("token", "")

    url = f"ws://localhost:{port}/kabusapi/websocket?filter=ALL"
    ws = create_connection(url, header=[f"X-API-KEY: {token}"], timeout=5)

    print("[INFO] connected to", url)
    while True:
        try:
            msg = ws.recv()
            if not msg:
                continue
            obj = json.loads(msg)

            if looks_like_tick(obj):
                print("[TICK]", json.dumps(obj, ensure_ascii=False) if not args.pretty else json.dumps(obj, ensure_ascii=False, indent=2))
            elif looks_like_board(obj):
                print("[BOARD]", json.dumps(obj, ensure_ascii=False) if not args.pretty else json.dumps(obj, ensure_ascii=False, indent=2))
            elif args.all:
                print("[RAW]", json.dumps(obj, ensure_ascii=False) if not args.pretty else json.dumps(obj, ensure_ascii=False, indent=2))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[ERROR]", e, file=sys.stderr)
            break

if __name__ == "__main__":
    main()
