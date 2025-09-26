import argparse
import csv
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
from scripts.common_config import load_json_utf8


@dataclass
class Config:
    universe_path: str
    perma_regulars_path: str
    manual_popular_path: str
    output_path: str
    max_output: int
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    random_seed: int
    log_level: str
    score_decimals: int


DEFAULT_CONFIG = {
    "paths": {
        "universe": "data/universe.csv",
        "perma_regulars": "data/perma_regulars.csv",
        "manual_popular": "data/manual_popular.csv",
        "output": "data/watchlist_today.csv",
    },
    "limits": {"max_output": 70},
    "weights": {
        "vol_surge": 40,
        "turnover": 30,
        "depth_stable": 20,
        "news_pop": 10,
        "manual_bonus": 10,
        "perma_bonus": 5,
    },
    "thresholds": {
        "vol_surge": 0.70,
        "turnover": 0.70,
        "depth_stable": 0.50,
        "news_pop": 0.50,
    },
    "random_seed": 42,
    "log_level": "INFO",
    "format": {
        "score_decimals": 2,
    },
}


def load_config(path: str) -> Config:
    cfg_data = DEFAULT_CONFIG
    if path and os.path.exists(path):
        try:
            loaded = load_json_utf8(path)
        except RuntimeError as e:
            print(f"Failed to parse config JSON: {e}", file=sys.stderr)
            sys.exit(1)
        cfg_data = {
            **cfg_data,
            **{k: (loaded.get(k, v)) for k, v in cfg_data.items()},
        }
        for section in ("paths", "limits", "weights", "thresholds", "format"):
            if section in loaded and isinstance(loaded[section], dict):
                cfg_data[section] = {**DEFAULT_CONFIG[section], **loaded[section]}

    paths = cfg_data.get("paths", {})
    limits = cfg_data.get("limits", {})
    weights = cfg_data.get("weights", {})

    return Config(
        universe_path=str(paths.get("universe", DEFAULT_CONFIG["paths"]["universe"])),
        perma_regulars_path=str(
            paths.get("perma_regulars", DEFAULT_CONFIG["paths"]["perma_regulars"])
        ),
        manual_popular_path=str(
            paths.get("manual_popular", DEFAULT_CONFIG["paths"]["manual_popular"])
        ),
        output_path=str(paths.get("output", DEFAULT_CONFIG["paths"]["output"])),
        max_output=int(limits.get("max_output", DEFAULT_CONFIG["limits"]["max_output"])),
        weights={
            "vol_surge": float(weights.get("vol_surge", DEFAULT_CONFIG["weights"]["vol_surge"])),
            "turnover": float(weights.get("turnover", DEFAULT_CONFIG["weights"]["turnover"])),
            "depth_stable": float(
                weights.get("depth_stable", DEFAULT_CONFIG["weights"]["depth_stable"])
            ),
            "news_pop": float(weights.get("news_pop", DEFAULT_CONFIG["weights"]["news_pop"])),
            "manual_bonus": float(
                weights.get("manual_bonus", DEFAULT_CONFIG["weights"]["manual_bonus"])
            ),
            "perma_bonus": float(
                weights.get("perma_bonus", DEFAULT_CONFIG["weights"]["perma_bonus"])
            ),
        },
        thresholds={
            "vol_surge": float(
                cfg_data.get("thresholds", {}).get("vol_surge", DEFAULT_CONFIG["thresholds"]["vol_surge"])
            ),
            "turnover": float(
                cfg_data.get("thresholds", {}).get("turnover", DEFAULT_CONFIG["thresholds"]["turnover"])
            ),
            "depth_stable": float(
                cfg_data.get("thresholds", {}).get("depth_stable", DEFAULT_CONFIG["thresholds"]["depth_stable"])
            ),
            "news_pop": float(
                cfg_data.get("thresholds", {}).get("news_pop", DEFAULT_CONFIG["thresholds"]["news_pop"])
            ),
        },
        random_seed=int(cfg_data.get("random_seed", DEFAULT_CONFIG["random_seed"])),
        log_level=str(cfg_data.get("log_level", DEFAULT_CONFIG["log_level"])).upper(),
        score_decimals=int(cfg_data.get("format", {}).get("score_decimals", DEFAULT_CONFIG["format"]["score_decimals"])),
    )


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


CODE_ALLOWED_RE = re.compile(r"^[0-9A-Za-z]+$")


def read_simple_csv(path: str, label: str) -> Tuple[List[Dict[str, str]], int, int]:
    """Read CSV with at least 'code' and optional 'name'.
    Returns: (records, total_rows_read)
    - Missing file or empty -> returns ([], 0) without raising.
    - Extra columns are ignored.
    - 'code' coerced to string; skip rows with empty code.
    - 'name' defaults to ''.
    """
    total = 0
    invalid = 0
    records: List[Dict[str, str]] = []
    if not path or not os.path.exists(path):
        return records, total, invalid
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            # Handle empty file (no header)
            if reader.fieldnames is None:
                logging.warning("Missing header treated as empty (%s)", label)
                return records, total, invalid
            if "code" not in [c.strip() for c in reader.fieldnames]:
                logging.warning("Missing required column 'code' treated as empty (%s)", label)
                return records, total, invalid
            for row in reader:
                total += 1
                code = str(row.get("code", "")).strip()
                if not code or not CODE_ALLOWED_RE.match(code):
                    invalid += 1
                    logging.warning("Invalid code skipped (%s): %s", label, code)
                    continue
                name = str(row.get("name", "")).strip()
                records.append({"code": code, "name": name})
    except Exception:
        # On any unexpected read error, treat as empty but log later by caller
        return [], total, invalid
    return records, total, invalid


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def compute_dummy_metrics(code: str, seed: int) -> Dict[str, float]:
    r = random.Random(f"{seed}:{code}")
    vol = r.random()  # [0,1)
    turn = r.random()  # [0,1)
    depth = 1.0 if r.random() < 0.5 else 0.0
    news = 1.0 if r.random() < 0.3 else 0.0
    return {
        "vol_surge": vol,
        "turnover": turn,
        "depth_stable": depth,
        "news_pop": news,
    }


def build_reason(flags: Dict[str, float], manual: bool, perma: bool, thresholds: Dict[str, float]) -> str:
    parts: List[str] = []
    if flags.get("vol_surge", 0) >= thresholds.get("vol_surge", 0.0):
        parts.append("vol_surge")
    if flags.get("turnover", 0) >= thresholds.get("turnover", 0.0):
        parts.append("turnover")
    if flags.get("depth_stable", 0) >= thresholds.get("depth_stable", 0.0):
        parts.append("depth_stable")
    if flags.get("news_pop", 0) >= thresholds.get("news_pop", 0.0):
        parts.append("news_pop")
    if manual:
        parts.append("manual")
    if perma:
        parts.append("perma")
    return ";".join(parts)


def generate_watchlist(cfg: Config, *, override_max: Optional[int] = None, override_seed: Optional[int] = None,
                       override_output: Optional[str] = None, dry_run: bool = False, preview_count: int = 10) -> int:
    logging.info("Start generate_watchlist")
    t0 = time.perf_counter()

    # Read inputs
    uni, uni_total, uni_invalid = read_simple_csv(cfg.universe_path, "universe")
    pr, pr_total, pr_invalid = read_simple_csv(cfg.perma_regulars_path, "perma_regulars")
    mp, mp_total, mp_invalid = read_simple_csv(cfg.manual_popular_path, "manual_popular")

    logging.info(
        "Read inputs: universe=%d/%d, perma_regulars=%d/%d, manual_popular=%d/%d",
        len(uni), uni_total, len(pr), pr_total, len(mp), mp_total,
    )

    # Merge with de-duplication (last one wins)
    merged: Dict[str, Dict[str, str]] = {}
    duplicates = 0
    order = [("universe", uni), ("perma_regulars", pr), ("manual_popular", mp)]
    for label, rows in order:
        for row in rows:
            code = row["code"]
            if code in merged:
                duplicates += 1
            merged[code] = {"code": code, "name": row.get("name", "")}

    logging.info("After merge: candidates=%d, duplicates_removed=%d", len(merged), duplicates)

    if len(merged) == 0:
        # Still need to write header-only CSV
        main_output = override_output or cfg.output_path
        top50_output = os.path.join(os.path.dirname(os.path.abspath(main_output)) or ".", "watchlist_top50.csv")
        if not dry_run:
            ensure_parent_dir(main_output)
            with open(main_output, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["rank", "code", "name", "score", "Reason"])
            ensure_parent_dir(top50_output)
            with open(top50_output, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["rank", "code", "name", "score", "Reason"])
            logging.info("No candidates. Wrote header-only outputs: %s, %s", main_output, top50_output)
        else:
            logging.info("No candidates. Dry-run: no files written.")
            _print_table([])
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        summary = (
            f"summary read={uni_total}/{pr_total}/{mp_total} "
            f"valid={len(uni)+len(pr)+len(mp)} dup={duplicates} out={0} elapsed_ms={elapsed_ms}"
        )
        logging.info(summary)
        logging.info("Completed generate_watchlist")
        return 0

    # Prepare sets for bonuses
    manual_set = {row["code"] for row in mp}
    perma_set = {row["code"] for row in pr}

    # Score
    scored: List[Dict[str, str]] = []
    seed = override_seed if override_seed is not None else cfg.random_seed
    for code, item in merged.items():
        metrics = compute_dummy_metrics(code, seed)
        score = (
            cfg.weights["vol_surge"] * metrics["vol_surge"]
            + cfg.weights["turnover"] * metrics["turnover"]
            + cfg.weights["depth_stable"] * metrics["depth_stable"]
            + cfg.weights["news_pop"] * metrics["news_pop"]
        )
        is_manual = code in manual_set
        is_perma = code in perma_set
        if is_manual:
            score += cfg.weights.get("manual_bonus", 0)
        if is_perma:
            score += cfg.weights.get("perma_bonus", 0)
        reason = build_reason(metrics, is_manual, is_perma, cfg.thresholds)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            dbg = {
                "vol": metrics["vol_surge"] >= cfg.thresholds.get("vol_surge", 0.0),
                "turn": metrics["turnover"] >= cfg.thresholds.get("turnover", 0.0),
                "depth": metrics["depth_stable"] >= cfg.thresholds.get("depth_stable", 0.0),
                "news": metrics["news_pop"] >= cfg.thresholds.get("news_pop", 0.0),
                "manual": is_manual,
                "perma": is_perma,
            }
            logging.debug(
                "flags code=%s vol=%s turn=%s depth=%s news=%s manual=%s perma=%s score=%.4f",
                code, int(dbg["vol"]), int(dbg["turn"]), int(dbg["depth"]), int(dbg["news"]), int(dbg["manual"]), int(dbg["perma"]), score,
            )
        scored.append(
            {
                "code": code,
                "name": item.get("name", ""),
                "score": score,
                "Reason": reason,
            }
        )

    logging.info("Scored %d candidates", len(scored))

    # Sort by score desc, tie-break by code asc
    scored.sort(key=lambda r: (-r["score"], r["code"]))
    # Apply override max if provided
    max_output = override_max if override_max is not None else cfg.max_output
    limited = scored[: max(0, int(max_output))]

    # Add rank (1-based)
    for idx, row in enumerate(scored, start=1):
        row["rank"] = idx
    for idx, row in enumerate(limited, start=1):
        row["rank"] = idx

    # Write output
    main_output = override_output or cfg.output_path
    top50_output = os.path.join(os.path.dirname(os.path.abspath(main_output)) or ".", "watchlist_top50.csv")

    if dry_run:
        # Pretty print table for first N rows
        n = min(max(0, int(preview_count)), int(max_output))
        preview = limited[:n]
        _print_table(preview, decimals=cfg.score_decimals)
        logging.info("Dry-run: no files written")
    else:
        ensure_parent_dir(main_output)
        with open(main_output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "code", "name", "score", "Reason"])
            for row in limited:
                writer.writerow([row["rank"], row["code"], row["name"], format(row['score'], f".{cfg.score_decimals}f"), row["Reason"]])

        # Top 50
        ensure_parent_dir(top50_output)
        # top50 must be the first 50 rows of today's list
        top50_rows = limited[:50]
        with open(top50_output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "code", "name", "score", "Reason"])
            for row in top50_rows:
                writer.writerow([row["rank"], row["code"], row["name"], format(row['score'], f".{cfg.score_decimals}f"), row["Reason"]])

        logging.info("Wrote outputs: %s (rows=%d), %s (rows=%d)", main_output, len(limited), top50_output, len(top50_rows))

    # Summary line
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    summary = (
        f"summary read={uni_total}/{pr_total}/{mp_total} "
        f"valid={len(uni)+len(pr)+len(mp)} dup={duplicates} out={len(limited)} elapsed_ms={elapsed_ms}"
    )
    logging.info(summary)
    logging.info("Completed generate_watchlist")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate watchlist from CSV sources")
    p.add_argument(
        "-Config",
        dest="config",
        default="config/stream_settings.json",
        help="Path to config JSON",
    )
    p.add_argument("--max", dest="max_output", type=int, default=None, help="Override max output rows")
    p.add_argument("--seed", dest="seed", type=int, default=None, help="Override random seed")
    p.add_argument("--output", dest="output", type=str, default=None, help="Override output CSV path for today list")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="Do not write files; print a table preview")
    p.add_argument("--debug", dest="debug", action="store_true", help="Enable DEBUG logging level")
    p.add_argument("--preview", dest="preview", type=int, default=10, help="Rows to preview with --dry-run (default 10)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    # Apply debug override
    if args.debug:
        setup_logging("DEBUG")
    else:
        setup_logging(cfg.log_level)
    try:
        rc = generate_watchlist(
            cfg,
            override_max=args.max_output,
            override_seed=args.seed,
            override_output=args.output,
            dry_run=bool(args.dry_run),
            preview_count=args.preview,
        )
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        sys.exit(1)
    sys.exit(rc)


def _print_table(rows: List[Dict[str, str]], *, decimals: int = 2) -> None:
    # Determine widths
    headers = ["rank", "code", "name", "score", "Reason"]
    str_rows = [
        {
            "rank": str(r.get("rank", "")),
            "code": str(r.get("code", "")),
            "name": str(r.get("name", "")),
            "score": format(r.get("score", 0) or 0, f".{decimals}f"),
            "Reason": str(r.get("Reason", "")),
        }
        for r in rows
    ]
    widths = {h: len(h) for h in headers}
    for r in str_rows:
        for h in headers:
            widths[h] = max(widths[h], len(r[h]))

    def fmt_row(r: Dict[str, str]) -> str:
        return (
            f"{r['rank']:<{widths['rank']}}  {r['code']:<{widths['code']}}  "
            f"{r['name']:<{widths['name']}}  {r['score']:>{widths['score']}}  {r['Reason']:<{widths['Reason']}}"
        )

    header_line = (
        f"{'rank':<{widths['rank']}}  {'code':<{widths['code']}}  "
        f"{'name':<{widths['name']}}  {'score':>{widths['score']}}  {'Reason':<{widths['Reason']}}"
    )
    sep_line = "-" * len(header_line)
    print(header_line)
    print(sep_line)
    for r in str_rows:
        print(fmt_row(r))


if __name__ == "__main__":
    main()
