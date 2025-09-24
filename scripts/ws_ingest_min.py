import argparse, json, sqlite3, datetime, time
from pathlib import Path
import websocket

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True)
    ap.add_argument("-Seconds", type=int, default=600)
    args = ap.parse_args()

    cfg = json.load(open(args.Config, "r", encoding="utf-8-sig"))
    port  = int(cfg.get("port", 18080))
    token = (cfg.get("token") or "").strip()
    db    = cfg["db_path"]

    ws = websocket.create_connection(f"ws://localhost:{port}/kabusapi/websocket",
                                     header=[f"X-API-KEY: {token}"], timeout=5)
    print("[WS] connected")

    con = sqlite3.connect(db, timeout=5)
    cur = con.cursor()
    # 既存テーブル想定。無ければ最小構成だけ作る
    cur.execute("""CREATE TABLE IF NOT EXISTS orderbook_snapshot(
        ts TEXT, bid1 REAL, ask1 REAL
    )""")
    con.commit()

    ws.settimeout(3)
    deadline = time.time() + args.Seconds
    n = 0
    while time.time() < deadline:
        try:
            _ = ws.recv()  # 受信できた事実だけ使う
            n += 1
            ts = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))\
                    .strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("INSERT INTO orderbook_snapshot(ts) VALUES (?)", (ts,))
            if n % 20 == 0:
                con.commit()
                print(f"[INGEST] inserted={n}, ts={ts}")
        except websocket.WebSocketTimeoutException:
            pass
        except Exception as e:
            print("[ERR]", e)
            break
    con.commit()
    con.close()
    ws.close()
    print(f"[DONE] total_inserts={n}")

if __name__ == "__main__":
    main()
