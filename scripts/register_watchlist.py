# scripts/register_watchlist.py
# kabuステーションAPI /register に watchlist_top50.csv の code を登録するスクリプト
# 使い方（PowerShell）:
#   py .\scripts\register_watchlist.py -Config .\config\stream_settings.json -Input .\data\watchlist_top50.csv
# オプション:
#   -Max 50（上限指定）  -DryRun（送信せず内容確認）  -Verbose 1

import argparse
import csv
from pathlib import Path
from scripts.common_config import load_json_utf8
import sys
import time
from typing import List, Dict

try:
    import requests
except ImportError:
    print(
        "requests が未インストールです。PowerShellで:  py -m pip install requests",
        file=sys.stderr,
    )
    sys.exit(1)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return load_json_utf8(path)


def load_codes(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # まずは試すエンコード候補
    encodings = ["utf-8-sig", "cp932", "utf-8", "latin-1"]

    last_err = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                # 区切り文字推定（, / ; / \t など）
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except Exception:
                    # 失敗したらカンマ前提
                    dialect = csv.excel
                reader = csv.DictReader(f, dialect=dialect)

                if reader.fieldnames is None:
                    raise RuntimeError("CSVヘッダが見つかりません")

                # 許容カラム名
                cols_lower = [c.lower() for c in reader.fieldnames]
                if "code" in cols_lower:
                    code_col = reader.fieldnames[cols_lower.index("code")]
                elif "ticker" in cols_lower:
                    code_col = reader.fieldnames[cols_lower.index("ticker")]
                elif "銘柄コード" in reader.fieldnames:
                    code_col = "銘柄コード"
                else:
                    raise RuntimeError(f"code列が見つかりません。ヘッダ: {reader.fieldnames}")

                codes: List[str] = []
                for row in reader:
                    code = (row.get(code_col) or "").strip()
                    if code:
                        codes.append(code)
                if not codes:
                    raise RuntimeError("CSVにcodeが1件もありません")
                return codes
        except Exception as e:
            last_err = e
            continue

    # すべて失敗
    raise last_err or RuntimeError("CSV読込に失敗しました")


def build_symbols(codes: List[str], exchange: int = 1) -> List[Dict[str, str]]:
    # kabu API 仕様に合わせ {"Symbol": "<code>", "Exchange": 1} を作る（東証=1）
    # 新規上場の英数字コード（例: 215A）もそのまま渡す
    return [{"Symbol": c, "Exchange": exchange} for c in codes]


def put_register(
    port: int, token: str, symbols: List[Dict[str, str]], verbose: int = 0
) -> requests.Response:
    url = f"http://localhost:{port}/kabusapi/register"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": token,
    }
    payload = {"Symbols": symbols}
    if verbose:
        print(f"[HTTP] PUT {url}")
        print(f"[HTTP] payload count={len(symbols)}")
    resp = requests.put(url, headers=headers, json=payload, timeout=10)
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Config", required=True, help="設定JSONパス（port, token を含む）")
    ap.add_argument("-Input", required=True, help="登録CSVパス（code列が必要）")
    ap.add_argument("-Max", type=int, default=50, help="登録上限（既定50）")
    ap.add_argument("-Exchange", type=int, default=1, help="取引所コード（東証=1）")
    ap.add_argument("-Verbose", type=int, default=1, help="冗長ログ 0/1")
    ap.add_argument("-DryRun", action="store_true", help="送信せずに内容だけ表示")
    args = ap.parse_args()

    config = load_config(Path(args.Config))
    port = int(config.get("port", 18080))
    token = (config.get("token") or "").strip()
    if not token:
        raise RuntimeError(
            "Configの token が空です。先に kabus_login_wait.py を実行してください。"
        )

    codes = load_codes(Path(args.Input))
    if args.Verbose:
        print(f"[WATCHLIST] loaded={len(codes)} from={args.Input}")

    # 上限適用（先頭からMax件）
    codes = codes[: max(1, args.Max)]
    symbols = build_symbols(codes, exchange=args.Exchange)

    print(
        f"[WATCHLIST] final={len(symbols)} (preview top 10): {[s['Symbol'] for s in symbols[:10]]}"
    )

    if args.DryRun:
        print("[REGISTER] DryRun=ON → API送信は実行しません。")
        return

    # PUT /register（本APIは上書き型：渡したリストが当日の登録集合になります）
    try:
        resp = put_register(port, token, symbols, verbose=args.Verbose)
        ok = resp.status_code == 200
        body = {}
        try:
            body = resp.json()
        except Exception:
            body = {"text": resp.text[:300]}
        print(
            f"[REGISTER] status={resp.status_code} ok={ok} body_keys={list(body.keys())}"
        )
        if not ok:
            print(
                f"[FAIL] source=register reason=status{resp.status_code} body={body}",
                file=sys.stderr,
            )
            sys.exit(2)
        print("[REGISTER] Completed. PUSH配信の対象が更新されました。")
    except requests.exceptions.RequestException as e:
        print(
            f"[FAIL] source=register reason=request_exception msg={e}", file=sys.stderr
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
