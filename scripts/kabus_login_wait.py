import argparse
import json
import time
import requests
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-Config", required=True, help="設定ファイルのパス (JSON)")
    args = parser.parse_args()

    config_path = Path(args.Config)
    conf = json.loads(config_path.read_text(encoding="utf-8"))

    port = conf.get("port", 18080)
    api_password = conf.get("api_password")
    if not api_password:
        raise RuntimeError("configに api_password がありません")

    url = f"http://localhost:{port}/kabusapi/token"
    payload = {"APIPassword": api_password}

    print(f"🔑 APIトークン要求中... ({url})")
    for i in range(10):  # 最大10回リトライ
        try:
            r = requests.post(url, json=payload)
            if r.status_code == 200:
                data = r.json()
                if "Token" in data:
                    conf["token"] = data["Token"]
                    config_path.write_text(
                        json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    print("✅ トークン取得成功:", data["Token"])
                    return
                else:
                    print("⚠️ 応答にTokenが含まれません:", data)
            else:
                print("⚠️ status:", r.status_code, r.text)
        except Exception as e:
            print("⚠️ Error:", e)

        time.sleep(3)

    raise RuntimeError("トークン取得に失敗しました")


if __name__ == "__main__":
    main()
