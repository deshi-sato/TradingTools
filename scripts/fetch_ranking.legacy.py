import argparse
import csv
import logging
import os
import sys
from scripts.common_config import load_json_utf8
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # fallback to avoid import error during static checks

# kabuステーションAPIのランキング種別は数値コード（仕様準拠）
CATEGORY_MAP = {
    "up": 1,        # 値上がり率
    "down": 2,      # 値下がり率
    "volume": 3,    # 売買高上位
    "turnover": 4,  # 売買代金
    "tick": 5,      # TICK回数
    "vol_surge": 6, # 売買高急増
    "to_surge": 7,  # 売買代金急増
    # 8〜15: 信用系・連騰系など
}

# 出力時のreason列に並べる順
REASON_TAGS = ["turnover", "volume", "up", "down"]

# 既定の抽出件数
CATEGORY_DEFAULT_LIMITS = {
    "turnover": 20,
    "volume": 20,
    "up": 10,
    "down": 10,
}

# MarketCode 互換 → ExchangeDivision 変換
EXDIV_MAP = {
    1:   "ALL",  # 全市場
    101: "TP",   # 東証プライム
    102: "TS",   # 東証スタンダード
    103: "TG",   # 東証グロース
    104: "M",    # 名証
    105: "FK",   # 福証
    106: "S",    # 札証
}


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> Dict[str, Any]:
    return load_json_utf8(path)


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def try_get(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalize_item(obj: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[int], Optional[float]]]:
    code = try_get(obj, ["Symbol", "symbol", "Code", "IssueCode"])
    name = try_get(obj, ["SymbolName", "Name", "IssueName"])
    rank = try_get(obj, ["Rank", "No"])
    score = try_get(obj, ["Score", "ChangeRate", "ChangePercentage", "Turnover", "Volume", "TickCount"])

    if not code or not name:
        return None

    try:
        code_s = str(code)
        name_s = str(name)
    except Exception:
        return None

    r_val: Optional[int] = None
    s_val: Optional[float] = None
    try:
        if rank is not None:
            r_val = int(rank)
    except Exception:
        pass
    try:
        if score is not None:
            s_val = float(score)
    except Exception:
        pass

    return code_s, name_s, r_val, s_val


def find_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if "Ranking" in payload and isinstance(payload["Ranking"], list):
            return payload["Ranking"]
    return []


def request_rankings(
    base_url: str,
    token: str,
    exchange_div: str,
    category: str,
    timeout: int,
) -> List[Tuple[str, str, Optional[int], Optional[float]]]:
    url = base_url.rstrip("/") + "/ranking"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-API-KEY": token,
    }
    params = {
        "Type": CATEGORY_MAP[category],
        "ExchangeDivision": exchange_div,
    }

    if requests is None:
        logging.error("requests module is not available")
        return []

    logging.debug("GET %s params=%s", url, params)
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        logging.debug("Response: %s", resp.status_code)
    except Exception as e:
        logging.error("HTTP GET failed: %s", e)
        return []

    if resp.status_code != 200:
        logging.error("API error: %s %s", resp.status_code, resp.text[:200])
        return []

    try:
        payload = resp.json()
    except Exception as e:
        logging.error("Invalid JSON: %s", e)
        return []

    items = find_items(payload)
    out: List[Tuple[str, str, Optional[int], Optional[float]]] = []
    for obj in items:
        norm = normalize_item(obj)
        if norm:
            code, name, rank, score = norm
            logging.debug("item: code=%s name=%s rank=%s score=%s", code, name, rank, score)
            out.append(norm)
    return out


def aggregate_rankings(
    base_url: str,
    token: str,
    market_code: int,
    timeout: int,
    per_category_limits: Dict[str, int],
) -> List[Tuple[str, str, List[str]]]:
    combined: Dict[str, Tuple[str, set]] = {}
    exchange_div = EXDIV_MAP.get(market_code, "ALL")

    for category, limit in per_category_limits.items():
        logging.info("Fetching %s (limit %d)", category, limit)
        items = request_rankings(base_url, token, exchange_div, category, timeout)
        if not items:
            logging.warning("No items returned for %s", category)
        for code, name, _rank, _score in items[: max(0, int(limit))]:
            if code not in combined:
                combined[code] = (name, set())
            old_name, reasons = combined[code]
            final_name = old_name if old_name else name
            reasons.add(category)
            combined[code] = (final_name, reasons)

    rows: List[Tuple[str, str, List[str]]] = []
    for code, (name, reasons) in combined.items():
        ordered = [r for r in REASON_TAGS if r in reasons] + [r for r in reasons if r not in REASON_TAGS]
        rows.append((code, name, ordered))

    return rows


def write_csv(path: str, rows: Iterable[Tuple[str, str, List[str]]]) -> None:
    ensure_parent_dir(path)
    # UTF-8 (BOM付き) で保存 → Excelでも文字化けせず開ける
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "reason"])
        for code, name, reasons in rows:
            w.writerow([code, name, ";".join(reasons)])


def print_csv(rows: Iterable[Tuple[str, str, List[str]]]) -> None:
    print("code,name,reason")
    for code, name, reasons in rows:
        print(f"{code},{name},{';'.join(reasons)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch rankings and export CSV")
    p.add_argument("-Config", required=True, help="Path to JSON config")
    p.add_argument("--limit", type=int, help="Override limit for all categories")
    p.add_argument("--output", help="Output CSV path (default from config)")
    p.add_argument("--dry-run", action="store_true", help="Print CSV to stdout only")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.debug)

    try:
        cfg = load_config(args.Config)
    except Exception as e:
        logging.error("Failed to read config: %s", e)
        return 1

    kabu = cfg.get("kabu", {})
    base_url = str(kabu.get("base_url", "")).strip()
    api_token = str(kabu.get("api_token", "")).strip()

    if not base_url or not api_token:
        logging.error("Config kabu.base_url and kabu.api_token are required")
        return 1

    ranking_cfg = cfg.get("ranking", {})
    market_code = int(ranking_cfg.get("market_code", 1))
    default_output = str(ranking_cfg.get("default_output", "data/perma_regulars.csv"))
    timeout_sec = int(ranking_cfg.get("timeouts_sec", 10))

    output_path = args.output or default_output

    if args.limit is not None:
        per_limits = {k: int(args.limit) for k in CATEGORY_DEFAULT_LIMITS}
    else:
        per_limits = dict(CATEGORY_DEFAULT_LIMITS)

    logging.info("API start: %s", base_url)
    rows = aggregate_rankings(
        base_url=base_url,
        token=api_token,
        market_code=market_code,
        timeout=timeout_sec,
        per_category_limits=per_limits,
    )
    logging.info("Fetched %d unique symbols", len(rows))

    if args.dry_run:
        print_csv(rows)
        return 0

    try:
        write_csv(output_path, rows)
        logging.info("Wrote CSV: %s", output_path)
    except Exception as e:
        logging.error("Failed to write CSV: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
