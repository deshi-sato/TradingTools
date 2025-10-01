# scripts/ws_print_ticker.py
# -*- coding: utf-8 -*-
"""
Kabu WebSocket からティック(Price/Volume/ExecutionDateTime)だけを受信して表示するビューア
設定は config/stream_settings.json を使用:
  - token   : X-API-KEY に使用
  - (任意) symbols はログ表示用。フィルタは --filter で指定
ws_url は固定: ws://localhost:18080/kabusapi/websocket
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, Optional, Set

try:
    from websocket import create_connection  # pip install websocket-client
except Exception as e:
    sys.stderr.write("websocket-client が未インストールです。`py -m pip install websocket-client`\n")
    raise

WS_URL = "ws://localhost:18080/kabusapi/websocket"


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json_utf8(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def looks_like_tick(payload: Dict[str, Any]) -> bool:
    # 典型的な Tick 形
    return all(k in payload for k in ("Symbol", "Price", "Volume", "ExecutionDateTime"))


def looks_like_board(payload: Dict[str, Any]) -> bool:
    return all(k in payload for k in ("Symbol", "BidPrice", "AskPrice"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Print ticks from Kabu WebSocket")
    ap.add_argument("-Config", required=True, help="path to stream_settings.json")
    ap.add_argument("--filter", default="", help="comma-separated symbols to print (例: 285A,9501)")
    ap.add_argument("--pretty", action="store_true", help="きれいに整形して表示")
    ap.add_argument("--timeout", type=float, default=2.0, help="ws recv timeout seconds")
    ap.add_argument("--all", action="store_true", help="tick 以外(Board等)も表示")
    args = ap.parse_args()

    cfg = load_json_utf8(args.Config)
    token = cfg.get("token") or cfg.get("kabu", {}).get("api_token")
    if not token:
        sys.stderr.write("[ERROR] stream_settings.json の 'token' が未設定です。\n")
        sys.exit(2)

    wanted: Set[str] = set(s.strip() for s in args.filter.split(",") if s.strip())

    print(f"[BOOT] url={WS_URL} filter={','.join(sorted(wanted)) or 'ALL'} timeout={args.timeout}")
    headers = [f"X-API-KEY: {token}"]

    while True:
        ws = None
        try:
            ws = create_connection(WS_URL, header=headers, timeout=args.timeout)
            print(f"[INFO] connected  {iso_now()}")
            while True:
                raw = ws.recv()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
                except Exception:
                    continue

                if not isinstance(payload, dict):
                    continue

                # シンボルでフィルタ
                sym = str(payload.get("Symbol") or "")
                if wanted and sym and sym not in wanted:
                    if not args.all:
                        continue

                if looks_like_tick(payload):
                    # 需要がある情報だけ抜粋
                    out = {
                        "ts": payload.get("ExecutionDateTime"),
                        "sym": sym,
                        "price": payload.get("Price"),
                        "vol": payload.get("Volume"),
                        "side": payload.get("BuySell"),  # 1:買 2:売 等のベンダ表記想定
                        "seq": payload.get("SeqNum"),
                    }
                    if args.pretty:
                        print(f"[TICK] {out['ts']} {out['sym']:>6} "
                              f"{out['price']:>8} x {out['vol']:<6} side={out['side']} seq={out['seq']}")
                    else:
                        print(json.dumps({"type": "tick", **out}, ensure_ascii=False))
                    continue

                if args.all and looks_like_board(payload):
                    out = {
                        "ts": payload.get("ExecutionDateTime") or payload.get("CurrentPriceTime"),
                        "sym": sym,
                        "bid": payload.get("BidPrice"),
                        "ask": payload.get("AskPrice"),
                        "bqty": payload.get("BidQty") or payload.get("TotalBidQty"),
                        "aqty": payload.get("AskQty") or payload.get("TotalAskQty"),
                        "seq": payload.get("SeqNum"),
                    }
                    if args.pretty:
                        print(f"[BOARD] {out['ts']} {out['sym']:>6} "
                              f"bid={out['bid']} ask={out['ask']} bqty={out['bqty']} aqty={out['aqty']} seq={out['seq']}")
                    else:
                        print(json.dumps({"type": "board", **out}, ensure_ascii=False))
        except KeyboardInterrupt:
            print("[INFO] stopped by user")
            break
        except Exception as e:
            print(f"[WARN] {e}; reconnecting...", flush=True)
            time.sleep(1.0)
        finally:
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
