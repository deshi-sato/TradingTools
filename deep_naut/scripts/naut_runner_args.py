import argparse
import os
from datetime import datetime


def today_ymd() -> str:
    return datetime.now().strftime("%Y%m%d")


def default_features_db(symbol: str) -> str:
    sym = (symbol or "").strip() or "UNKNOWN"
    return os.path.join("db", f"naut_features_{sym}_{today_ymd()}.db")


def default_ops_db(symbol: str) -> str:
    sym = (symbol or "").strip() or "UNKNOWN"
    return os.path.join("db", f"naut_ops_{sym}_{today_ymd()}.db")


def default_log_path(symbol: str) -> str:
    sym = (symbol or "").strip() or "runner"
    return os.path.join("logs", f"naut_runner_{sym}_{today_ymd()}.log")


def primary_symbol(symbols) -> str:
    if not symbols:
        return ""
    if isinstance(symbols, (list, tuple)):
        first = symbols[0] if symbols else ""
        return str(first).split(",")[0].strip()
    return str(symbols).split(",")[0].strip()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="naut_runner", description="Deep Naut runner bootstrap CLI"
    )
    ap.add_argument("--symbols", help="Symbol list (comma separated)")
    ap.add_argument("--symbol", help="Single symbol alias for --symbols")
    ap.add_argument(
        "--mode", choices=["AUTO", "BUY", "SELL"], default="AUTO", help="Execution mode"
    )
    ap.add_argument("--thr", dest="thr", help="Threshold JSON (best_thresholds)")
    ap.add_argument("--broker", choices=["paper", "live"], default="paper")
    ap.add_argument("--dry-run", type=int, choices=[0, 1], default=1)
    ap.add_argument("--config", default="deep_naut/config/runner_settings.json")
    ap.add_argument("--policy", default="")
    ap.add_argument("--feature-source", choices=["features_stream", "raw_push"])
    ap.add_argument("--features-db", dest="features_db")
    ap.add_argument("--raw-db", dest="raw_db", help="Alias of --features-db (raw_push habit)")
    ap.add_argument("--ops-db", dest="ops_db")
    ap.add_argument("--log-path", dest="log_path")
    ap.add_argument("--verbose", type=int, choices=[0, 1], default=0)
    ap.add_argument("--replay-from-start", action="store_true")
    ap.add_argument("--window", help="JST window HH:MM[-HH:MM] or HH:MM:SS[-HH:MM:SS]")
    # ML options
    ap.add_argument("--ml-model")
    ap.add_argument("--ml-feat-names")
    ap.add_argument("--prob-up-len", type=int, default=3)
    ap.add_argument("--vol-ma3-thr", type=float, default=700.0)
    ap.add_argument("--vol-rate-thr", type=float, default=1.30)
    ap.add_argument("--vol-gate", choices=["OR", "AND"], default="OR")
    ap.add_argument("--sync-ticks", type=int, default=3)
    ap.add_argument("--cooldown-ms", type=int, default=1500)
    return ap


def finalize_defaults(args):
    if getattr(args, "symbol", None) and not getattr(args, "symbols", None):
        args.symbols = [str(args.symbol)]
    sym = primary_symbol(getattr(args, "symbols", ""))
    if getattr(args, "raw_db", None) and not getattr(args, "features_db", None):
        args.features_db = args.raw_db
    if not getattr(args, "features_db", None):
        args.features_db = default_features_db(sym)
    if not getattr(args, "ops_db", None):
        args.ops_db = default_ops_db(sym)
    if not getattr(args, "log_path", None):
        args.log_path = default_log_path(sym)
    return args
