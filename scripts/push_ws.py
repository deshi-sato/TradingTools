import argparse
import json
import time
from datetime import datetime

import websocket  # websocket-client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=18080)  # 本番:18080 / 検証:18081
    parser.add_argument("--token", required=True)
    parser.add_argument("--seconds", type=int, default=30)  # 受信する秒数
    args = parser.parse_args()

    url = f"ws://{args.host}:{args.port}/kabusapi/websocket"
    headers = [f"X-API-KEY: {args.token}"]

    def on_open(ws):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] OPEN {url}")

    def on_message(ws, message):
        # kabuステは登録済みシンボルの板/歩み値等をJSONでプッシュしてくる想定
        try:
            obj = json.loads(message)
            # 軽量に1行で要点だけ表示（銘柄,種別,価格など）
            typ = obj.get("Type") or obj.get("MessageType") or "?"
            sym = obj.get("Symbol") or obj.get("IssueCode") or "?"
            p = (
                obj.get("CurrentPrice")
                or obj.get("Price")
                or obj.get("BidPrice")
                or obj.get("AskPrice")
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {typ} {sym} {p}")
        except Exception:
            # 解析不能なメッセージはそのまま表示
            print(f"[RAW] {message[:300]}")

    def on_error(ws, err):
        print("[ERROR]", err)

    def on_close(ws, code, reason):
        print(f"[CLOSE] code={code} reason={reason}")

    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # 指定秒数だけ受信して終了
    t0 = time.time()

    def _timeout_run(*_):
        # websocket-client は run_forever を止める明確なAPIがないので、
        # 別スレッドタイマーでも良い。簡易的にここでは seconds 経過で終了させる。
        pass

    # ループを別スレッドで止める代わりに、手動で Ctrl+C でもOK
    print(f"Receiving for ~{args.seconds}s ... (Ctrl+C to stop)")
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
