# -*- coding: utf-8 -*-
"""
ML breakout gate utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MlGateConfig:
    prob_up_len: int = 3
    vol_ma3_thr: float = 700.0
    vol_rate_thr: float = 1.30
    use_and: bool = False
    sync_ticks: int = 3
    cooldown_ms: int = 1500


class MlBreakoutGate:
    """prob↑ × (vol_ma3, vol_rate) × candle_up の同期検知ゲート"""

    def __init__(self, cfg: MlGateConfig):
        self.cfg = cfg
        self._pbuf: List[float] = []
        self._last_hit_ms: Optional[int] = None
        self._last_idx: Optional[int] = None

    def reset(self) -> None:
        self._pbuf.clear()
        self._last_hit_ms = None
        self._last_idx = None

    def _prob_up(self) -> bool:
        k = max(1, int(self.cfg.prob_up_len))
        buf = self._pbuf
        if len(buf) < k:
            return False
        for i in range(k - 1):
            if buf[i] >= buf[i + 1]:
                return False
        return True

    def _vol_ok(self, feat: Dict) -> bool:
        v3 = float(feat.get("vol_ma3") or 0.0)
        vr = float(feat.get("vol_rate") or 0.0)
        v3_ok = v3 >= float(self.cfg.vol_ma3_thr)
        vr_ok = vr >= float(self.cfg.vol_rate_thr)
        return (v3_ok and vr_ok) if self.cfg.use_and else (v3_ok or vr_ok)

    def check(self, feat: Dict) -> Optional[Dict]:
        """
        feat には最低限:
          idx, t_exec(ms), pstar(prob), candle_up(±1 or 0),
          vol_ma3(float or None), vol_rate(float or None)
        """
        if feat is None:
            return None
        idx = int(feat.get("idx") or 0)
        t_exec = int(feat.get("t_exec") or 0)
        prob = float(feat.get("pstar") or 0.0)
        candle_up = float(feat.get("candle_up") or 0.0)
        up_ok = candle_up > 0.0

        if self._last_hit_ms is not None:
            delta = t_exec - self._last_hit_ms
            if delta < int(self.cfg.cooldown_ms):
                self._append_prob(prob)
                return None

        self._append_prob(prob)

        vol_ok = self._vol_ok(feat)
        hit = False
        reason = ""
        if up_ok and vol_ok and self._prob_up():
            hit = True
            reason = "prob_up_vol"
        elif up_ok and vol_ok and self._last_idx is not None:
            if idx - self._last_idx <= int(self.cfg.sync_ticks):
                hit = True
                reason = "sync_prob_lag"

        if hit:
            self._last_hit_ms = t_exec
            self._last_idx = idx
            return {
                "idx": idx,
                "t_exec": t_exec,
                "pstar": prob,
                "vol_ma3": feat.get("vol_ma3"),
                "vol_rate": feat.get("vol_rate"),
                "reason": reason,
            }
        return None

    def _append_prob(self, value: float) -> None:
        k = max(1, int(self.cfg.prob_up_len))
        self._pbuf.append(float(value))
        if len(self._pbuf) > k:
            self._pbuf = self._pbuf[-k:]
