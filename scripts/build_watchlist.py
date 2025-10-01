# build_watchlist.py
from __future__ import annotations
import argparse, csv, logging, sys, re, glob, os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

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
# 正規化：code/name を抽出
# ----------------------------
def normalize_to_code_name(rows: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    入力行から code（= 証券コード / シンボル相当）, name（銘柄名）を抽出。
    name が無い場合は空文字にする。
    """
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for row in rows:
        code = (
            row.get("code")
            or row.get("Code")
            or row.get("symbol")
            or row.get("Symbol")
            or ""
        ).strip()
        if not code:
            continue

        # そのまま大文字化のみ（新規上場の英数字コード 215A 等にも対応）
        code_norm = code.upper()

        # 銘柄名の候補キー
        name = (
            row.get("name")
            or row.get("Name")
            or row.get("銘柄名")
            or row.get("company")
            or ""
        ).strip()

        if code_norm in seen:
            continue
        seen.add(code_norm)
        out.append((code_norm, name))
    return out

# ----------------------------
# 書き出し：rank, code, name, score, reason
# ----------------------------
def write_watchlist_top50(path: Path, items: List[Tuple[str, str]], fixed_score: int = 50) -> None:
    """
    watchlist_top50 と同じ5カラムで出力。
    - rank: 1始まりの連番（入力順）
    - code: 正規化済みコード
    - name: 可能なら入力から採用。無ければ空文字
    - score: 常に fixed_score（デフォルト 50）
    - reason: 空文字（要件が出たら差し替え可能）
    """
    ensure_parent(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "code", "name", "score", "reason"])
        for idx, (code, name) in enumerate(items, start=1):
            w.writerow([idx, code, name, fixed_score, ""])

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
    ap = argparse.ArgumentParser(description="Build today's watchlist (top50-format output)")
    ap.add_argument("--data-dir", default=r".\data", help="perma/fallback の格納ディレクトリ")
    ap.add_argument("--output",   default=r".\data\watchlist_today.csv", help="出力CSV（rank,code,name,score,reason）")
    ap.add_argument("--limit",    type=int, help="上位N件までに制限（例: 50）")
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
        fp, ts = _latest_by_glob(str(data_dir / "fallback_daytrade_*.csv"))
        if not fp:
            logging.error("No fallback_daytrade_*.csv in: %s", data_dir)
            return 1
        source = Path(fp); picked = f"fallback_daytrade ({ts})"
    else:
        source, picked = choose_latest_source(data_dir)

    logging.info("[watchlist] ranking source = %s -> %s", picked, source)

    rows = read_csv_any(source)
    items = normalize_to_code_name(rows)
    if not items:
        logging.error("Selected source is empty or invalid: %s", source)
        return 1

    if args.limit is not None and args.limit >= 0:
        items = items[: args.limit]

    write_watchlist_top50(out_path, items, fixed_score=50)
    logging.info("Wrote watchlist(top50-format): %s (%d rows) [source=%s]", out_path, len(items), picked)
    return 0

if __name__ == "__main__":
    sys.exit(main())
