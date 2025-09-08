#!/usr/bin/env python3
"""
Benchmark baseline script.

Spec:
- Count total bars from dataset_1min/*.csv (sum of rows per file, excluding header).
- Provide run_baseline_backtest(dataset_dir, strategy_cfg) as a placeholder.
- After execution, print JSON to stdout with keys:
  elapsed_sec, bars, bars_per_sec, rss_mem_mb, result
- Use psutil for memory measurement and time.perf_counter for timing.
- Example strategy_cfg: {"ma_fast": 5, "ma_slow": 25}
"""

from __future__ import annotations

import csv
import glob
import json
import os
import time
from typing import Dict, Any

import psutil


def count_bars(dataset_dir: str) -> int:
    pattern = os.path.join(dataset_dir, "*.csv")
    total = 0
    for path in glob.glob(pattern):
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                first = True
                for row in reader:
                    if first:
                        first = False
                        # Skip header if it looks like one
                        if row and row[0].strip().lower() == "timestamp":
                            continue
                    # Count data row
                    if row:
                        total += 1
        except FileNotFoundError:
            continue
    return total


def run_baseline_backtest(dataset_dir: str, strategy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for the actual backtest runner.

    Replace this with your current backtester entry point. Keep the signature.
    Return whatever summary you want to embed under "result" in the final JSON.
    """
    # Dummy result for now
    symbols = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(dataset_dir, "*.csv"))]
    return {
        "status": "ok",
        "strategy": strategy_cfg,
        "symbols": symbols,
        "note": "placeholder backtest",
    }


def main() -> None:
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "dataset_1min"))
    strategy_cfg = {"ma_fast": 5, "ma_slow": 25}

    # Pre-count bars to report throughput relative to total dataset size
    bars = count_bars(dataset_dir)

    t0 = time.perf_counter()
    result = run_baseline_backtest(dataset_dir, strategy_cfg)
    elapsed = time.perf_counter() - t0

    # RSS in MiB
    proc = psutil.Process(os.getpid())
    rss_bytes = proc.memory_info().rss
    rss_mem_mb = rss_bytes / (1024 * 1024)

    bars_per_sec = (bars / elapsed) if elapsed > 0 else 0.0

    out = {
        "elapsed_sec": round(elapsed, 6),
        "bars": int(bars),
        "bars_per_sec": round(bars_per_sec, 3),
        "rss_mem_mb": round(rss_mem_mb, 3),
        "result": result,
    }

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

