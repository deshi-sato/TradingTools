\"\"\"generate_report.py : Auto-generated placeholder

- file: generate_report.py
- updated: 2025-09-08

TODO: このモジュールの概要をここに書いてください。
\"\"\"
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv as _csv


def normalize_key(s: str) -> str:
    return ''.join(ch.lower() for ch in s if ch.isalnum())


STRATEGY_KEYS = {"strategy", "name", "model", "signal", "label", "tag"}
SHARPE_ALIASES = {"sharpe", "sharperatio", "sharpe_ratio"}
CUM_ALIASES = {
    "cumulativereturn",
    "cumulative",
    "cumreturn",
    "totalreturn",
    "returncumulative",
    "cumend",
}
MAXDD_ALIASES = {"maxdd", "maxdrawdown", "maxdrawdownratio"}


@dataclass
class Metrics:
    sharpe: Optional[float] = None
    cum_return: Optional[float] = None  # as decimal, e.g., 0.25 for 25%
    max_dd: Optional[float] = None      # as negative decimal, e.g., -0.18 for -18%
    monthly_winrate: Optional[float] = None  # 0..1
    monthly_mean: Optional[float] = None     # as decimal


def parse_float(s: str) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if t == "" or t.lower() in {"na", "nan", "none", "null", "-"}:
        return None
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    try:
        if t.endswith("%"):
            v = float(t[:-1].replace(",", "").strip()) / 100.0
            return -v if neg else v
        v = float(t.replace(",", ""))
        return -v if neg else v
    except ValueError:
        return None


def _sniff_dialect(path: Path) -> Optional[_csv.Dialect]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
        dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect
    except Exception:
        return None


def read_summary(path: Path) -> Dict[str, Metrics]:
    if not path.exists():
        return {}
    dialect = _sniff_dialect(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = _csv.reader(f, dialect=dialect) if dialect else _csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}

    header = [normalize_key(h) for h in rows[0]]

    # Tall format detection: has an explicit strategy/name column
    has_strategy_col = any(h in STRATEGY_KEYS for h in header)
    metrics_by_strategy: Dict[str, Metrics] = {}

    if has_strategy_col:
        # DictReader with original headers for value extraction
        with path.open(newline='', encoding='utf-8-sig') as f:
            dr = _csv.DictReader(f, dialect=dialect) if dialect else _csv.DictReader(f)
            for row in dr:
                # find strategy name
                strategy = None
                for k in row.keys():
                    if normalize_key(k) in STRATEGY_KEYS:
                        strategy = (row.get(k) or "").strip()
                        break
                if not strategy:
                    continue
                m = metrics_by_strategy.setdefault(strategy, Metrics())
                for k, v in row.items():
                    nk = normalize_key(k)
                    if nk in SHARPE_ALIASES:
                        pv = parse_float(v)
                        if pv is not None:
                            m.sharpe = pv
                    elif nk in CUM_ALIASES:
                        pv = parse_float(v)
                        if pv is not None:
                            # cum_end in our pipeline is equity multiple; convert to return
                            if nk == "cumend":
                                m.cum_return = pv - 1.0
                            else:
                                # Assume CSV might store as % already; keep as decimal
                                m.cum_return = pv
                    elif nk in MAXDD_ALIASES:
                        pv = parse_float(v)
                        if pv is not None:
                            # Store as decimal; ensure sign is negative if value is positive magnitude
                            if pv > 0:
                                pv = -abs(pv)
                            m.max_dd = pv
        return metrics_by_strategy

    # Wide format detection: first column is metric names, others are strategies
    # e.g., Row1: Metric, StratA, StratB ; Row2: Sharpe, 1.2, 0.8 ; Row3: CumulativeReturn, 0.35, 0.27
    strategies = rows[0][1:]
    for strat in strategies:
        metrics_by_strategy[strat] = Metrics()
    for row in rows[1:]:
        if not row:
            continue
        metric_name = normalize_key(row[0])
        for i, strat in enumerate(strategies, start=1):
            val = row[i] if i < len(row) else None
            pv = parse_float(val)
            if pv is None:
                continue
            m = metrics_by_strategy[strat]
            if metric_name in SHARPE_ALIASES:
                m.sharpe = pv
            elif metric_name in CUM_ALIASES:
                # For wide-format, assume values already represent returns (not equity multiple)
                m.cum_return = pv
            elif metric_name in MAXDD_ALIASES:
                if pv > 0:
                    pv = -abs(pv)
                m.max_dd = pv
    return metrics_by_strategy


def read_monthly_table(path: Path) -> Tuple[List[str], Dict[str, List[float]], Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    if not path.exists():
        return [], {}, {}, {}
    dialect = _sniff_dialect(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader_obj = _csv.reader(f, dialect=dialect) if dialect else _csv.reader(f)
        rows = list(reader_obj)
    if not rows:
        return [], {}, {}, {}

    raw_headers = rows[0]
    headers_n = [normalize_key(h) for h in raw_headers]

    # Vertical monthly format: has 'tag' plus 'win_rate' and/or 'mean_ret'
    if ("tag" in headers_n or any(h in STRATEGY_KEYS for h in headers_n)) and (
        "winrate" in headers_n or "meanret" in headers_n
    ):
        with path.open(newline='', encoding='utf-8-sig') as f:
            dr = _csv.DictReader(f, dialect=dialect) if dialect else _csv.DictReader(f)
            order: List[str] = []
            monthly_returns: Dict[str, List[float]] = {}
            agg_win: Dict[str, List[float]] = {}
            for row in dr:
                # find tag/strategy
                strategy = None
                for k in row.keys():
                    if normalize_key(k) in STRATEGY_KEYS:
                        strategy = (row.get(k) or "").strip()
                        break
                if not strategy:
                    continue
                if strategy not in order:
                    order.append(strategy)
                # collect per-row values
                mr = row.get("mean_ret") if "mean_ret" in row else row.get("mean ret")
                if mr is None:
                    # try normalized key lookup
                    for k in row.keys():
                        if normalize_key(k) == "meanret":
                            mr = row[k]
                            break
                wr = row.get("win_rate") if "win_rate" in row else row.get("win rate")
                if wr is None:
                    for k in row.keys():
                        if normalize_key(k) == "winrate":
                            wr = row[k]
                            break
                mrv = parse_float(mr) if mr is not None else None
                wrv = parse_float(wr) if wr is not None else None
                if mrv is not None:
                    monthly_returns.setdefault(strategy, []).append(mrv)
                if wrv is not None:
                    agg_win.setdefault(strategy, []).append(wrv)
        strategies = order
        # aggregate
        agg_winrate: Dict[str, Optional[float]] = {}
        agg_meanret: Dict[str, Optional[float]] = {}
        for s in strategies:
            wins = agg_win.get(s, [])
            agg_winrate[s] = (sum(wins) / len(wins)) if wins else None
            rets = monthly_returns.get(s, [])
            agg_meanret[s] = (sum(rets) / len(rets)) if rets else None
        return strategies, monthly_returns, agg_winrate, agg_meanret

    # Horizontal monthly format (wide): first column is date/month label, others are strategies with monthly returns
    strategies = raw_headers[1:]
    cols_raw: Dict[str, List[str]] = {s: [] for s in strategies}
    for r in rows[1:]:
        if not r:
            continue
        for idx, s in enumerate(strategies, start=1):
            cols_raw[s].append(r[idx] if idx < len(r) else "")

    def detect_and_parse(col_vals: List[str]) -> List[Optional[float]]:
        parsed = [parse_float(x) for x in col_vals]
        # Decide scale: if many values > 1 and <= 100, treat as percent numbers
        numeric = [x for x in parsed if x is not None]
        if not numeric:
            return [None if x is None else x for x in parsed]
        gt1 = sum(1 for x in numeric if abs(x) > 1.0)
        le100 = sum(1 for x in numeric if abs(x) <= 100.0)
        if gt1 > 0 and gt1 / max(1, len(numeric)) >= 0.2 and le100 == len(numeric):
            scale = 0.01
        else:
            scale = 1.0
        return [None if x is None else x * scale for x in parsed]

    monthly_returns: Dict[str, List[float]] = {}
    agg_winrate: Dict[str, Optional[float]] = {}
    agg_meanret: Dict[str, Optional[float]] = {}
    for s in strategies:
        parsed_vals = detect_and_parse(cols_raw[s])
        vals = [x for x in parsed_vals if x is not None]
        monthly_returns[s] = vals
        if vals:
            agg_meanret[s] = sum(vals) / len(vals)
            # Approximate monthly winrate as fraction of positive months
            agg_winrate[s] = sum(1 for x in vals if x > 0) / len(vals)
        else:
            agg_meanret[s] = None
            agg_winrate[s] = None
    return strategies, monthly_returns, agg_winrate, agg_meanret


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    n = len(vals)
    if n == 0:
        return None, None
    mu = sum(vals) / n
    if n < 2:
        return mu, None
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return mu, var ** 0.5


def compute_sharpe(monthly: List[float], periods_per_year: int = 12) -> Optional[float]:
    mu, sd = mean_std(monthly)
    if mu is None or sd is None or sd == 0:
        return None
    import math
    return (mu / sd) * math.sqrt(periods_per_year)


def compute_cum(monthly: List[float]) -> Optional[float]:
    if not monthly:
        return None
    total = 1.0
    for r in monthly:
        total *= (1.0 + r)
    return total - 1.0


def compute_maxdd(monthly: List[float]) -> Optional[float]:
    if not monthly:
        return None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0  # as negative number
    for r in monthly:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        dd = (equity / peak) - 1.0  # <= 0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def pct(x: Optional[float], digits: int = 1, sign: bool = False) -> str:
    if x is None:
        return "-"
    val = x * 100.0
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}%"
    return fmt.format(val)


def fmt_float(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def build_report(
    summary_csv: Path,
    monthly_csv: Path,
    images: List[Path],
    out_md: Path,
    title: str,
) -> None:
    summary = read_summary(summary_csv)
    strategies_from_summary = list(summary.keys())
    strategies_from_monthly, monthly_returns, agg_winrate, agg_meanret = read_monthly_table(monthly_csv)
    # Union of strategies preserving order: first summary, then monthly
    strategies: List[str] = []
    seen = set()
    for s in strategies_from_summary + strategies_from_monthly:
        if s and s not in seen:
            strategies.append(s)
            seen.add(s)

    # Compute monthly-based metrics where needed
    combined: Dict[str, Metrics] = {}
    for s in strategies:
        m = Metrics()
        if s in summary:
            sm = summary[s]
            m.sharpe = sm.sharpe
            m.cum_return = sm.cum_return
            m.max_dd = sm.max_dd
        if s in monthly_returns:
            series = monthly_returns[s]
            if series:
                # Monthly win rate and mean
                # Prefer aggregated values if provided (vertical format), else derive from series
                m.monthly_winrate = agg_winrate.get(s)
                m.monthly_mean = agg_meanret.get(s)
                if m.monthly_winrate is None:
                    wins = sum(1 for x in series if x > 0)
                    m.monthly_winrate = wins / len(series)
                if m.monthly_mean is None:
                    mu, _ = mean_std(series)
                    m.monthly_mean = mu
                # Backfill missing headline metrics
                if m.sharpe is None:
                    m.sharpe = compute_sharpe(series)
                if m.cum_return is None:
                    m.cum_return = compute_cum(series)
                if m.max_dd is None:
                    m.max_dd = compute_maxdd(series)
        combined[s] = m

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"生成時刻: {ts}")
    lines.append("")
    lines.append("## 主要指標サマリー")
    lines.append("")
    # Header
    lines.append("| 戦略 | Sharpe | 累積リターン | 最大DD | 月次勝率 | 月次平均リターン |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in strategies:
        m = combined.get(s, Metrics())
        sharpe_s = fmt_float(m.sharpe, 2)
        cum_s = pct(m.cum_return, 2)
        maxdd_s = pct(m.max_dd, 1)
        win_s = pct(m.monthly_winrate, 1)
        mean_s = pct(m.monthly_mean, 2, sign=True)
        lines.append(f"| {s} | {sharpe_s} | {cum_s} | {maxdd_s} | {win_s} | {mean_s} |")

    # Images section
    if images:
        lines.append("")
        lines.append("## グラフ")
        lines.append("")
        for img in images:
            rel = img.as_posix()
            # Alt text based on file stem
            alt = img.stem
            lines.append(f"![{alt}]({rel})")
            lines.append("")

    out_md.write_text("\n".join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description="Generate Markdown report from comparison CSVs.")
    p.add_argument("--summary", default="compare_summary.csv", type=Path, help="Path to summary CSV")
    p.add_argument("--monthly", default="compare_monthly_table.csv", type=Path, help="Path to monthly table CSV")
    p.add_argument("--out", default="report.md", type=Path, help="Output Markdown file path")
    p.add_argument("--title", default="パフォーマンス比較レポート", help="Report title")
    p.add_argument(
        "--images",
        nargs="*",
        type=Path,
        default=[
            Path("compare_cum.png"),
            Path("compare_monthly_winrate.png"),
            Path("compare_monthly_mean.png"),
        ],
        help="List of image paths to embed",
    )
    args = p.parse_args()

    build_report(args.summary, args.monthly, args.images, args.out, args.title)
    print(f"Markdown report written to: {args.out}")


if __name__ == "__main__":
    main()
