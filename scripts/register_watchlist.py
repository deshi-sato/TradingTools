# scripts/register_watchlist.py
# kabuステーションAPI /register に watchlist_top50.csv の code を登録し、
# stream_settings.json の "symbols" と "price_guard" を更新する。
# 追加修正: 登録前に /unregister/all を必ず実行し、結果をコンソール表示する。

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict

try:
    import requests
except ImportError:
    print("requests が未インストールです。PowerShellで:  py -m pip install requests", file=sys.stderr)
    sys.exit(1)


def load_json_utf8(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json_utf8(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_codes(csv_path: Path) -> List[str]:
    encodings = ["utf-8-sig", "cp932", "utf-8", "latin-1"]
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                cols = [c.lower() for c in reader.fieldnames or []]
                if "code" in cols:
                    col = reader.fieldnames[cols.index("code")]
                elif "ticker" in cols:
                    col = reader.fieldnames[cols.index("ticker")]
                elif "銘柄コード" in reader.fieldnames:
                    col = "銘柄コード"
                else:
                    raise RuntimeError("CSVに code 列がありません")

                codes = [row[col].strip() for row in reader if row.get(col)]
                if codes:
                    return codes
        except Exception:
            continue
    raise RuntimeError("CSV読込に失敗しました")


def build_symbols(codes: List[str], exchange: int = 1) -> List[Dict[str, str]]:
    return [{"Symbol": c, "Exchange": exchange} for c in codes]


def put_register(port: int, token: str, symbols: List[Dict[str, str]]) -> requests.Response:
    url = f"http://localhost:{port}/kabusapi/register"
    headers = {"Content-Type": "application/json", "X-API-KEY": token}
    payload = {"Symbols": symbols}
    return requests.put(url, headers=headers, json=payload, timeout=10)


def fetch_price_guard(port: int, token: str, code: str, exchange: int = 1) -> Dict[str, float]:
    url = f"http://localhost:{port}/kabusapi/symbol/{code}@{exchange}"
    headers = {"X-API-KEY": token}
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"/symbol/{code} status={resp.status_code}")
    data = resp.json()
    return {
        "min": float(data.get("LowerLimit") or 0),
        "max": float(data.get("UpperLimit") or 0),
    }


def unregister_all(port: int, token: str) -> None:
    """全解除を実行して結果をコンソールに表示"""
    url = f"http://localhost:{port}/kabusapi/register/all"
    headers = {"X-API-KEY": token}
    try:
        resp = requests.delete(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            print("[UNREGISTER] all symbols cleared")
        else:
            print(f"[UNREGISTER] failed status={resp.status_code} body={resp.text}", file=sys.stderr)
    except Exception as e:
        print(f"[UNREGISTER] exception: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True, help="設定JSONパス（port, token を含む）")
    ap.add_argument("-Input", required=True, help="登録CSVパス（code列が必要）")
    ap.add_argument("-Max", type=int, default=50)
    ap.add_argument("-Exchange", type=int, default=1)
    ap.add_argument("-Verbose", type=int, default=1)
    ap.add_argument("-DryRun", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.Config)
    config = load_json_utf8(config_path)
    port = int(config.get("port", 18080))
    token = (config.get("token") or "").strip()
    if not token:
        raise RuntimeError("Configの token が空です。先に kabus_login_wait.py を実行してください。")

    # --- 追加: 全解除 ---
    unregister_all(port, token)

    codes = load_codes(Path(args.Input))[: max(1, args.Max)]
    symbols = build_symbols(codes, exchange=args.Exchange)

    if args.Verbose:
        print(f"[WATCHLIST] loaded {len(codes)} codes: {codes[:10]}...")

    if args.DryRun:
        print("[REGISTER] DryRun=ON → API送信しません。")
    else:
        resp = put_register(port, token, symbols)
        if resp.status_code != 200:
            print(f"[FAIL] register status={resp.status_code} body={resp.text}", file=sys.stderr)
            sys.exit(2)
        print("[REGISTER] Completed.")

    # --- stream_settings.json 更新 ---
    config["symbols"] = codes
    pg_map: Dict[str, Dict[str, float]] = {}
    for c in codes:
        try:
            pg_map[c] = fetch_price_guard(port, token, c, args.Exchange)
            if args.Verbose:
                print(f"[PRICE_GUARD] {c} -> {pg_map[c]}")
        except Exception as e:
            print(f"[WARN] {c} price_guard取得失敗: {e}", file=sys.stderr)
    config["price_guard"] = pg_map

    save_json_utf8(config_path, config)
    print(f"[CONFIG] Updated {config_path} symbols+price_guard")


if __name__ == "__main__":
    main()
