import argparse
import json
import requests
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Fetch kabuステーション API token and update config"
    )
    p.add_argument(
        "-Config",
        required=True,
        help="Path to config JSON (e.g. config/stream_settings.json)",
    )
    p.add_argument("--pass", dest="api_pass", required=True, help="本番APIパスワード")
    args = p.parse_args()

    config_path = Path(args.Config)

    # 設定読み込み
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 1

    base_url = cfg.get("kabu", {}).get("base_url", "").rstrip("/")
    if not base_url:
        print("configに base_url がありません")
        return 1

    # トークン発行
    url = base_url + "/token"
    try:
        res = requests.post(url, json={"APIPassword": args.api_pass}, timeout=5)
    except Exception as e:
        print(f"HTTP error: {e}")
        return 1

    if not res.ok:
        print(f"API error {res.status_code}: {res.text[:200]}")
        return 1

    token = res.json().get("Token")
    if not token:
        print("Token not found in response")
        return 1

    # 設定更新
    cfg["kabu"]["api_token"] = token
    config_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"新しいトークンを config に保存しました: {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
