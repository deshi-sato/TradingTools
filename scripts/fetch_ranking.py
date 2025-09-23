from __future__ import annotations
import os, sys, csv, json, time, re
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Tuple, List

# ---- 可能なら exec.api_client を利用、無ければローカルHTTPでフォールバック
HAVE_CLIENT = False
try:
    from exec.api_client import get_ranking as _client_get_ranking  # type: ignore
    HAVE_CLIENT = True
except Exception:
    HAVE_CLIENT = False

import requests  # pip install requests

BASE = os.environ.get("KABU_BASE_URL", "http://localhost:18080").rstrip("/")
TIMEOUT = float(os.environ.get("KABU_HTTP_TIMEOUT_POST", "8"))

# 既定：東証の「売買代金20 / 出来高20 / 値上がり10 / 値下がり10」
RANK_PLAN = [
    ("trading_value", 1, 20, 1),  # (reasonタグ, Type, 件数, ExchangeDivision)
    ("volume",        2, 20, 1),
    ("top_gainers",   3, 10, 1),
    ("top_losers",    4, 10, 1),
]

def _issue_token_from_env() -> str:
    pw = os.environ.get("KABU_API_PW") or os.environ.get("KABU_API_PASSWORD")
    if not pw:
        return ""
    url = f"{BASE}/kabusapi/token"
    r = requests.post(url, headers={"Content-Type":"application/json"},
                      data=json.dumps({"APIPassword": pw}), timeout=TIMEOUT)
    r.raise_for_status()
    tok = r.json().get("Token") or ""
    if tok:
        os.environ["KABU_TOKEN"] = tok
        os.environ["KABU_API_KEY"] = tok
    return tok

def _get_ranking_direct(rtype: int, exchange_division: int, count: int):
    url = f"{BASE}/kabusapi/ranking"
    headers = {"Content-Type": "application/json"}
    tok = os.environ.get("KABU_TOKEN") or os.environ.get("KABU_API_KEY")
    if tok:
        headers["X-API-KEY"] = tok

    # 取引所は文字列指定（ALL/T/T1/T2 等）。既定=ALL
    exch = os.environ.get("RANK_EXCHANGE", "ALL")
    params = {"Type": int(rtype), "ExchangeDivision": exch}

    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    if r.status_code == 401 and _issue_token_from_env():
        headers["X-API-KEY"] = os.environ.get("KABU_TOKEN") or os.environ.get("KABU_API_KEY") or ""
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    jd = r.json() or {}
    items = jd.get("Ranking") or jd.get("ranking") or jd
    out = []
    if isinstance(items, list):
        for it in items[:count]:
            sym = str(it.get("Symbol") or it.get("symbol") or "").strip()
            ex = it.get("Exchange") or it.get("exchange") or 1
            try: ex = int(ex)
            except: ex = 1
            if sym:
                out.append((sym, ex))
    return out

def get_ranking(rtype: int, exchange_division: int, count: int) -> List[Tuple[str,int]]:
    if HAVE_CLIENT:
        try:
            return _client_get_ranking(rtype, exchange_division, count)  # type: ignore
        except Exception:
            pass
    return _get_ranking_direct(rtype, exchange_division, count)

def dedup_rankings(items: Iterable[Tuple[str,int,str]]):
    """重複銘柄は reason タグを統合（;区切り）"""
    tags = defaultdict(set)
    for sym, ex, why in items:
        tags[(sym, ex)].add(why)
    # 出現理由が多い順→銘柄コード順
    order = sorted(tags.items(), key=lambda kv: (-len(kv[1]), kv[0][0]))
    for (sym, ex), reasons in order:
        yield sym, ex, ";".join(sorted(reasons))

def fetch_plan():
    rows = []
    for tag, rtype, cnt, exch in RANK_PLAN:
        try:
            lst = get_ranking(rtype=rtype, exchange_division=exch, count=cnt)
        except Exception as e:
            print(f"[fetch_ranking] error {tag}: {e}", file=sys.stderr)
            lst = []
        for sym, ex in lst:
            rows.append((sym, ex, tag))
    return rows

def _now_ts12() -> str:
    """YYYYMMDDHHMM（ローカルタイム）"""
    return time.strftime("%Y%m%d%H%M")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fetch ranking and write perma_regulars_YYYYMMDDHHMM.csv")
    ap.add_argument("--outdir",   default="data", help="出力ディレクトリ（CSVは履歴として蓄積）")
    ap.add_argument("--encoding", default="utf-8-sig", help="CSVエンコーディング")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = fetch_plan()
    if not rows:
        print("[fetch_ranking] no data; nothing written", file=sys.stderr)
        sys.exit(2)  # 失敗扱い→watchlist 側で fallback

    uniq = list(dedup_rankings(rows))
    ts = _now_ts12()  # APIに更新時刻は無いので現在時刻で付番
    out_csv = outdir / f"perma_regulars_{ts}.csv"

    with out_csv.open("w", encoding=args.encoding, newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "exchange", "reason"])
        w.writerows(uniq)

    print(f"wrote {out_csv} ({len(uniq)} symbols)")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[fetch_ranking] fatal: {e}", file=sys.stderr)
        sys.exit(1)
