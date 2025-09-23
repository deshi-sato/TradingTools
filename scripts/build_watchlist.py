# build_watchlist.py
from __future__ import annotations
import argparse, csv, logging, sys, re, glob, os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
            exi = 1
        out.append((sym, exi))
    return out

# ----------------------------
# 書き出し：symbol,exchange 固定
# ----------------------------
def write_watchlist(path: Path, symbols: List[Tuple[str, int]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "exchange"])
        for s, ex in symbols:
            w.writerow([s, ex])

# ----------------------------
# 最新ファイル選択（YYYYMMDDHHMM をファイル名から抽出）
# ----------------------------
_TS12_RE = re.compile(r"(\d{12})")

def _latest_by_glob(pattern: str) -> Tuple[Optional[str], Optional[str]]:
    cands = []
    for p in glob.glob(pattern):
        m = _TS12_RE.search(os.path.basename(p))
        if m:
            cands.append((p, m.group(1)))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[1])  # 文字列比較でOK
    return cands[-1]  # (path, ts)

def choose_latest_source(data_dir: Path) -> Tuple[Path, str]:
    perma_path, perma_ts = _latest_by_glob(str(data_dir / "perma_regulars_*.csv"))
    fall_path,  fall_ts  = _latest_by_glob(str(data_dir / "fallback_daytrade_*.csv"))

    if not perma_ts and not fall_ts:
        raise FileNotFoundError("perma_regulars_*.csv / fallback_daytrade_*.csv が見つかりません。")

    if perma_ts and not fall_ts:
        return Path(perma_path), f"perma_regulars ({perma_ts})"
    if fall_ts and not perma_ts:
        return Path(fall_path), f"fallback_daytrade ({fall_ts})"

    # 両方ある → TS比較で新しい方を採用
    if perma_ts >= fall_ts:
        return Path(perma_path), f"perma_regulars ({perma_ts})"
    else:
        return Path(fall_path), f"fallback_daytrade ({fall_ts})"

# ----------------------------
# 引数
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build today's watchlist with latest perma/fallback")
    ap.add_argument("--data-dir", default=r".\data", help="perma/fallback の格納ディレクトリ")
    ap.add_argument("--output",   default=r".\data\watchlist_today.csv", help="出力CSV")
    ap.add_argument("--limit",    type=int, help="上位N件までに制限")
    ap.add_argument("--force-fallback", action="store_true", help="常に最新の fallback を採用（perma を無視）")
    ap.add_argument("--debug", action="store_true", help="DEBUGログを出す")
    return ap.parse_args()

# ----------------------------
# 本体
# ----------------------------
def main() -> int:
    args = parse_args()
    setup_logging(args.debug)

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)

    if args.force_fallback:
        # フォース時は fallback のみを探索
        fp, ts = _latest_by_glob(str(data_dir / "fallback_daytrade_*.csv"))
        if not fp:
            logging.error("No fallback_daytrade_*.csv in: %s", data_dir)
            return 1
        source = Path(fp); picked = f"fallback_daytrade ({ts})"
    else:
        source, picked = choose_latest_source(data_dir)

    logging.info("[watchlist] ranking source = %s -> %s", picked, source)

    rows = read_csv_any(source)
    syms = normalize_to_symbol(rows)
    if not syms:
        logging.error("Selected source is empty or invalid: %s", source)
        return 1

    if args.limit is not None and args.limit >= 0:
        syms = syms[: args.limit]

    write_watchlist(out_path, syms)
    logging.info("Wrote watchlist: %s (%d rows) [source=%s]", out_path, len(syms), picked)
    return 0

if __name__ == "__main__":
    sys.exit(main())
