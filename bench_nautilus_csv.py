#!/usr/bin/env python3
"""
CSV replay benchmark in a nautilus-like style.

Spec:
- Read dataset_1min/*.csv, ensure rows are sorted by timestamp.
- Stream close prices into a SimpleMACross (fast=5, slow=25) computed via numpy arrays.
- Output JSON to stdout with: engine, elapsed_sec, bars, bars_per_sec, rss_mem_mb, trades.

PowerShell example:
  python .\bench_nautilus_csv.py | Out-File -Encoding utf8 .\result_nautilus.json
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import psutil


ENGINE_NAME = "nautilus_like_csv_replay"


def friendly_exit(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def load_close_series(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load timestamp and close arrays from a CSV file.

    Returns
    - ts: numpy datetime64[s] array
    - close: float64 array
    """
    ts: List[str] = []
    close: List[float] = []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header_checked = False
            for row in r:
                if not row:
                    continue
                if not header_checked:
                    header_checked = True
                    # Skip header if present
                    if row[0].strip().lower() == "timestamp":
                        continue
                # Expect columns: timestamp,open,high,low,close,volume,symbol
                try:
                    ts.append(row[0].strip())
                    close.append(float(row[4]))
                except (IndexError, ValueError):
                    # Skip malformed rows
                    continue
    except FileNotFoundError:
        return np.array([], dtype="datetime64[s]"), np.array([], dtype=np.float64)

    if not ts:
        return np.array([], dtype="datetime64[s]"), np.array([], dtype=np.float64)

    ts_arr = np.array(ts, dtype="datetime64[s]")
    close_arr = np.asarray(close, dtype=np.float64)

    # Ensure sorted by timestamp
    order = np.argsort(ts_arr)
    if not np.all(order == np.arange(order.size)):
        ts_arr = ts_arr[order]
        close_arr = close_arr[order]

    return ts_arr, close_arr


def simple_ma_cross_trades(close: np.ndarray, fast: int = 5, slow: int = 25) -> int:
    """Count number of bullish crossovers (fast SMA crossing above slow SMA).

    Uses vectorized numpy convolutions; returns count of cross-up events as trade entries.
    """
    if close.size < slow or fast <= 0 or slow <= 0 or fast > slow:
        return 0

    # Simple moving averages via convolution
    w_fast = np.ones(fast, dtype=np.float64) / float(fast)
    w_slow = np.ones(slow, dtype=np.float64) / float(slow)
    sma_fast = np.convolve(close, w_fast, mode="valid")  # len = n - fast + 1
    sma_slow = np.convolve(close, w_slow, mode="valid")  # len = n - slow + 1

    # Align arrays to the same tail length where both defined
    offset = slow - fast
    if offset > 0:
        sma_fast = sma_fast[offset:]
    # Now lengths match: L = n - slow + 1
    L = sma_slow.shape[0]
    if sma_fast.shape[0] != L or L == 0:
        return 0

    diff = sma_fast - sma_slow
    sign = np.sign(diff)
    # Treat 0 as previous sign to avoid counting flat-to-positive as a cross if prev was positive
    # Shifted sign for previous step
    prev = np.empty_like(sign)
    prev[0] = 0
    prev[1:] = sign[:-1]
    # Cross up: prev <= 0 and current > 0
    cross_up = (prev <= 0) & (sign > 0)
    return int(np.count_nonzero(cross_up))


def main() -> None:
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "dataset_1min"))
    if not os.path.isdir(dataset_dir):
        friendly_exit(f"Dataset directory not found: {dataset_dir}")

    pattern = os.path.join(dataset_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        friendly_exit("No CSV files found in dataset_1min")

    fast, slow = 5, 25

    t0 = time.perf_counter()

    total_bars = 0
    total_trades = 0

    for path in files:
        ts_arr, close_arr = load_close_series(path)
        if close_arr.size == 0:
            continue
        total_bars += int(close_arr.size)
        total_trades += simple_ma_cross_trades(close_arr, fast=fast, slow=slow)

    elapsed = time.perf_counter() - t0

    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)

    bars_per_sec = (total_bars / elapsed) if elapsed > 0 else 0.0

    out = {
        "engine": ENGINE_NAME,
        "elapsed_sec": round(elapsed, 6),
        "bars": int(total_bars),
        "bars_per_sec": round(bars_per_sec, 3),
        "rss_mem_mb": round(rss_mb, 3),
        "trades": int(total_trades),
    }

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

