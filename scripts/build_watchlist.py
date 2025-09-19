import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    # 文字コードは UTF-8 と cp932 を順に試す
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append(
                        {
                            k.strip(): (v or "").strip()
                            for k, v in row.items()
                            if k is not None
                        }
                    )
            return rows
        except Exception as e:
            logging.debug("Decode retry with %s (%s)", enc, e)
    logging.error("Failed to read CSV: %s", path)
    return []


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_perma(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        code = (row.get("code") or row.get("Code") or "").strip()
        name = (row.get("name") or row.get("Name") or "").strip()
        reason = (row.get("reason") or row.get("Reason") or "").strip()
        if not code or not name:
            continue
        out.append(
            {"code": code, "name": name, "reason": reason or "perma", "source": "perma"}
        )
    return out


def normalize_fallback(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        code = (row.get("code") or row.get("Code") or "").strip()
        name = (row.get("name") or row.get("Name") or "").strip()
        reason = (row.get("reason") or row.get("Reason") or "fallback").strip()
        if not code or not name:
            continue
        out.append({"code": code, "name": name, "reason": reason, "source": "fallback"})
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "reason", "source"])
        for r in rows:
            w.writerow([r["code"], r["name"], r["reason"], r["source"]])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build today's watchlist with fallback")
    ap.add_argument(
        "-Perma",
        default=r".\data\perma_regulars.csv",
        help="Path to perma_regulars.csv",
    )
    ap.add_argument(
        "--fallback",
        default=r".\data\fallback_daytrade_core.csv",
        help="Path to fallback watchlist csv",
    )
    ap.add_argument(
        "--output", default=r".\data\watchlist_today.csv", help="Output csv path"
    )
    ap.add_argument("--limit", type=int, help="Limit number of rows")
    ap.add_argument(
        "--force-fallback", action="store_true", help="Force using fallback list"
    )
    ap.add_argument("--debug", action="store_true", help="Enable debug logs")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.debug)

    perma_path = Path(args.Perma)
    fallback_path = Path(args.fallback)
    out_path = Path(args.output)

    rows: List[Dict[str, str]] = []
    source_used = "fallback"

    if not args.force_fallback:
        perma = normalize_perma(read_csv(perma_path))
        if perma:
            rows = perma
            source_used = "perma"
            logging.info("Use perma list: %s (%d rows)", perma_path, len(perma))
        else:
            logging.warning("Perma list missing or empty: %s", perma_path)

    if not rows:
        fallback = normalize_fallback(read_csv(fallback_path))
        if not fallback:
            logging.error("Fallback list missing or empty: %s", fallback_path)
            return 1
        rows = fallback
        source_used = "fallback"
        logging.info("Use fallback list: %s (%d rows)", fallback_path, len(fallback))

    # 先頭から limit 件に丸める（順序は入力CSVの並び順）
    if args.limit is not None and args.limit >= 0:
        rows = rows[: args.limit]

    write_csv(out_path, rows)
    logging.info(
        "Wrote watchlist: %s (%d rows) [source=%s]", out_path, len(rows), source_used
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
