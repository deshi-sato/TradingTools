from typing import List, Tuple, Optional


def top3_sum(levels: Optional[List[Tuple[float, int]]]) -> int:
    if not levels:
        return 0
    return int(sum(q for _, q in levels[:3]))


def spread_bp(bid1: Optional[float], ask1: Optional[float]) -> Optional[float]:
    if bid1 is None or ask1 is None:
        return None
    mid = (bid1 + ask1) / 2.0
    if mid <= 0:
        return None
    return ((ask1 - bid1) / mid) * 10000.0


def depth_imbalance(buy_top3: int, sell_top3: int) -> float:
    denom = max(1, buy_top3 + sell_top3)
    return (buy_top3 - sell_top3) / denom


def uptick_ratio(upticks: int, downticks: int) -> float:
    denom = max(1, upticks + downticks)
    return upticks / denom
