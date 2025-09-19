"""Signal scanning primitives for intraday execution utilities."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal, Sequence

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    ts: int  # epoch milliseconds
    last: float
    vol: int


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    side: Side
    reason: str
    strength: float  # percent move versus lookback average
    last: float
    stop: float | None
    take: float | None


def scan_signals(ticks: Sequence[Tick]) -> list[Signal]:
    """Return breakout/breakdown signals based on the last 60 ticks.

    A BUY is emitted when the most recent price is at least +0.3%% above
    the rolling average; SELL when at least -0.3%% below. Stops/takes are left
    for the executor layer to enrich later.
    """
    if not ticks:
        return []

    window = [t for t in ticks[-60:] if t.last > 0]
    if not window:
        return []

    prices = [t.last for t in window]
    avg = fmean(prices)
    last = prices[-1]

    if avg <= 0:
        return []

    out: list[Signal] = []
    ratio = last / avg
    if ratio >= 1.003:
        strength = (ratio - 1.0) * 100.0
        out.append(
            Signal(
                symbol=window[-1].symbol,
                side="BUY",
                reason="avg_breakout",
                strength=strength,
                last=last,
                stop=None,
                take=None,
            )
        )
    if ratio <= 0.997:
        strength = (1.0 - ratio) * 100.0
        out.append(
            Signal(
                symbol=window[-1].symbol,
                side="SELL",
                reason="avg_breakdown",
                strength=strength,
                last=last,
                stop=None,
                take=None,
            )
        )
    return out
