# build_watchlist.py
from __future__ import annotations
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# ----------------------------
# ログ
# ----------------------------
def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

# ----------------------------
# CSV 読み込み（BOM/CP932 対応）
# ----------------------------
def read_csv_any(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    # None キー対策 & 余白除去
                    rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
            return rows
        except Exception as e:
            logging.debug("Decode retry with %s (%s)", enc, e)
    logging.error("Failed to read CSV: %s", path)
    return []

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 正規化：symbol/exchange へ寄せる
#   - perma も fallback も同じ正規化関数でOK
#   - 入力の列名ゆらぎに対応：
#       symbol | code
#       exchange | exch | market
#       name/reason は無視（無くてもOK）
# ----------------------------
def normalize_to_symbol(rows: List[Dict[str, str]]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for row in rows:
        sym = (row.get("symbol") or row.get("code") or row.get("Code") or "").strip()
        if not sym:
            continue
        ex = (row.get("exchange") or row.get("exch") or row.get("market") or "").strip()
        try:
            exi = int(ex) if ex else 1
        except Exception:
            # 文字コードや市場コード(JQ/T等)が来た時は既定=1
            exi = 1
        out.append((sym, exi))
    return out

# ----------------------------
# 書き出し：symbol,exchange 固定
# ----------------------------
def write_watchlist(path: Path, symbols: List[Tuple[str, int]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "exchange"])
        for s, ex in symbols:
            w.writerow([s, ex])

# ----------------------------
# 引数
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build today's watchlist with fallback (symbol,exchange)")
    ap.add_argument("-Perma", default=r".\data\perma_regulars.csv", help="Path to perma_regulars.csv")
    ap.add_argument("--fallback", default=r".\data\fallback_daytrade_core.csv", help="Path to fallback list csv")
    ap.add_argument("--output", default=r".\data\watchlist_today.csv", help="Output csv path")
    ap.add_argument("--limit", type=int, help="Limit number of rows")
    ap.add_argument("--force-fallback", action="store_true", help="Force using fallback list")
    ap.add_argument("--debug", action="store_true", help="Enable debug logs")
    return ap.parse_args()

# ----------------------------
# 本体
# ----------------------------
def main() -> int:
    args = parse_args()
    setup_logging(args.debug)

    perma_path = Path(args.Perma)
    fallback_path = Path(args.fallback)
    out_path = Path(args.output)

    picked = "fallback"
    symbols: List[Tuple[str, int]] = []

    if not args.force_fallback:
        perma_rows = read_csv_any(perma_path)
        perma_syms = normalize_to_symbol(perma_rows)
        if perma_syms:
            symbols = perma_syms
            picked = "perma"
            logging.info("Use perma list: %s (%d rows)", perma_path, len(perma_syms))
        else:
            logging.warning("Perma list missing or empty (or columns mismatch): %s", perma_path)

    if not symbols:
        fb_rows = read_csv_any(fallback_path)
        fb_syms = normalize_to_symbol(fb_rows)
        if not fb_syms:
            logging.error("Fallback list missing or empty: %s", fallback_path)
            return 1
        symbols = fb_syms
        picked = "fallback"
        logging.info("Use fallback list: %s (%d rows)", fallback_path, len(fb_syms))

    if args.limit is not None and args.limit >= 0:
        symbols = symbols[: args.limit]

    write_watchlist(out_path, symbols)
    logging.info("Wrote watchlist: %s (%d rows) [source=%s]", out_path, len(symbols), picked)
    return 0

if __name__ == "__main__":
    sys.exit(main())
