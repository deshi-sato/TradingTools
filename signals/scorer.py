# signals/scorer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

# signals/scorer.py の先頭付近にある想定
# 例）未導入なら追加
SCORE_LONG_THRESH  = int(os.environ.get("SCORE_LONG_THRESH", "7"))
SCORE_SHORT_THRESH = int(os.environ.get("SCORE_SHORT_THRESH", "7"))

@dataclass
class Score:
    total: int
    parts: Dict[str, int]
    side: Optional[str]  # "BUY" / "SELL" / None

def _pct(a: float, b: float) -> float:
    return (a - b) / b if b else 0.0

def _vol_in_window(cum_pairs: List[Tuple[int, int]], ms_from: int, ms_to: int) -> int:
    """
    cum_pairs: [(ts_ms, cumVol), ...] （時系列順）
    区間 [ms_from, ms_to] の出来高（累積差分）を返す。該当がなければ 0。
    """
    if not cum_pairs or ms_from >= ms_to:
        return 0
    # 二分探索するほどでもないので線形で十分（Nは小さい想定）
    v_from = None
    v_to = None
    for t, v in cum_pairs:
        if t <= ms_from:
            v_from = v
        if t <= ms_to:
            v_to = v
        else:
            break
    if v_from is None:
        v_from = cum_pairs[0][1]
    if v_to is None:
        v_to = cum_pairs[-1][1]
    return max(0, v_to - v_from)

def score_long(
    prev_high: Optional[float], prev_low: Optional[float], prev_close: Optional[float],
    today_high: float, today_low: float, last: float,
    vol_today: int,  # 当日累積出来高（終日累積）
    vol5d_avg: Optional[int],
    vol_10m: int, vol_prev30m: int,
    hh_like: bool, hl_like: bool,
) -> Score:
    parts: Dict[str, int] = {}

    # 1) トレンド（高値/安値の切り上げ）
    parts["trend"] = 2 if (hh_like and hl_like) else (1 if (hh_like or hl_like) else 0)

    # 2) 出来高の加速（10分 vs 直前30分）
    acc = 2 if (vol_10m >= vol_prev30m * 1.2 and vol_prev30m > 0) else \
          (1 if (vol_10m >= vol_prev30m * 1.05 and vol_prev30m > 0) else 0)
    parts["volume_accel"] = acc

    # 3) ブレイク（既定は前日高、無ければ当日高の更新性）
    ref_high = prev_high if prev_high else today_high
    br = _pct(last, ref_high)
    parts["break"] = 2 if br >= 0.005 else (1 if abs(br) < 0.005 else 0)

    # 4) 引け位置の代替（高値近接）
    near_hi = _pct(today_high, last)  # last が高値に近いほど値が小さい
    parts["closepos_like"] = 2 if near_hi <= 0.005 else (1 if near_hi <= 0.01 else 0)

    # 5) ボラ（当日高安レンジ / 参照終値）
    ref_close = prev_close if prev_close else last
    vola = (today_high - today_low) / ref_close if ref_close else 0.0
    parts["volatility_bonus"] = 1 if vola >= 0.03 else 0

    # 6) 出来高水準（当日累積 vs 5日平均）
    parts["vol_level_bonus"] = 1 if (vol5d_avg and vol_today >= vol5d_avg * 1.5) else 0

    total = sum(parts.values())
    side = "BUY" if (total >= 7 and parts["break"] == 2 and acc >= 1) else None
    return Score(total, parts, side)

def score_short(
    prev_high: Optional[float], prev_low: Optional[float], prev_close: Optional[float],
    today_high: float, today_low: float, last: float,
    vol_today: int, vol5d_avg: Optional[int],
    vol_10m: int, vol_prev30m: int,
    lh_like: bool, ll_like: bool, gapdown_open: bool
) -> Score:
    parts: Dict[str, int] = {}

    parts["trend"] = 2 if (lh_like and ll_like) else (1 if (lh_like or ll_like) else 0)

    # “天井後の失速”っぽさ（当日 < 直前期）
    parts["volume_decel"] = 2 if (vol_prev30m > 0 and vol_10m <= vol_prev30m * 0.8) else \
                            (1 if (vol_prev30m > 0 and vol_10m < vol_prev30m) else 0)

    # ブレイク（前日安、なければ当日安更新）
    ref_low = prev_low if prev_low else today_low
    under = _pct(last, ref_low)
    parts["break"] = 2 if (gapdown_open or under <= -0.005) else (1 if abs(under) < 0.005 else 0)

    # 安値近接
    near_lo = _pct(last, today_low)  # last が安値に近いほど小さい/負側
    parts["closepos_like"] = 2 if (-near_lo) <= 0.005 else (1 if (-near_lo) <= 0.01 else 0)

    ref_close = prev_close if prev_close else last
    vola = (today_high - today_low) / ref_close if ref_close else 0.0
    parts["volatility_bonus"] = 1 if vola >= 0.03 else 0

    parts["vol_level_bonus"] = 1 if (vol5d_avg and vol_today >= vol5d_avg * 1.5) else 0

    total = sum(parts.values())
    side = "SELL" if (total >= 7 and parts["break"] == 2) else None
    return Score(total, parts, side)
