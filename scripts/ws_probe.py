# scripts/ws_probe.py
import argparse, json, time
from pathlib import Path
import websocket  # websocket-client

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Seconds", type=int, default=15)
    ap.add_argument("-Verbose", type=int, default=1)
    args = ap.parse_args()

    cfg = json.load(open(args.Config, "r", encoding="utf-8-sig"))
    port  = int(cfg.get("port", 18080))
    token = (cfg.get("token") or "").strip()
    if not token:
        raise SystemExit("Config.token が空です。kabus_login_wait を先に実行してください。")

    url = f"ws://localhost:{port}/kabusapi/websocket"
    hdrs = [f"X-API-KEY: {token}"]

    if args.Verbose:
        print(f"[WS] connecting to {url}")

    ws = websocket.create_connection(url, header=hdrs, timeout=5)
    print("[WS] connected")

    ws.settimeout(3)
    deadline = time.time() + args.Seconds
    n = 0
    while time.time() < deadline:
        try:
            msg = ws.recv()
            n += 1
            # そのまま出すと長いので先頭だけ要約
            preview = (msg[:200] + "...") if len(msg) > 200 else msg
            print(f"[MSG {n}] {preview}")
        except websocket.WebSocketTimeoutException:
            print("[WS] (timeout waiting message)")
        except Exception as e:
            print(f"[WS] error: {e}")
            break
    ws.close()
    print(f"[WS] done. total_msgs={n}")

if __name__ == "__main__":
    main()
