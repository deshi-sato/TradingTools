from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class MlGateConfig:
    prob_up_len: int = 3
    vol_ma3_thr: float = 700.0
    vol_rate_thr: float = 1.30
    use_and: bool = False
    sync_ticks: int = 3
    cooldown_ms: int = 1500

class MlBreakoutGate:
    def __init__(self, cfg: MlGateConfig):
        self.cfg = cfg
        self._pbuf: List[float] = []
        self._last_hit_ms: Optional[int] = None
        self._last_idx: Optional[int] = None

    def reset(self):
        self._pbuf.clear()
        self._last_hit_ms = None
        self._last_idx = None

    def _prob_up(self) -> bool:
        k = self.cfg.prob_up_len
        b = self._pbuf
        return len(b) == k and all(b[i] < b[i+1] for i in range(k-1))

    def check(self, feat: Dict) -> Optional[Dict]:
        t = int(feat["t_exec"])
        p = float(feat["pstar"])
        up_ok = (feat.get("candle_up") or 0) > 0

        if self._last_hit_ms is not None and t - self._last_hit_ms < self.cfg.cooldown_ms:
            self._pbuf = (self._pbuf + [p])[-self.cfg.prob_up_len:]
            return None

        self._pbuf = (self._pbuf + [p])[-self.cfg.prob_up_len:]

        v3_ok = (feat.get("vol_ma3") or 0) >= self.cfg.vol_ma3_thr
        vr_ok = (feat.get("vol_rate") or 0) >= self.cfg.vol_rate_thr
        vol_ok = (v3_ok and vr_ok) if self.cfg.use_and else (v3_ok or vr_ok)

        hit = False
        reason = ""

        if up_ok and vol_ok and self._prob_up():
            hit, reason = True, "prob & vol & up"

        if not hit and up_ok and vol_ok and feat.get("idx") is not None:
            if self._last_idx is None or feat["idx"] - self._last_idx <= self.cfg.sync_ticks:
                hit, reason = True, "sync(prob lag)"

        if hit:
            self._last_hit_ms = t
            self._last_idx = feat.get("idx")
            return {
                "t_exec": t, "pstar": p,
                "vol_ma3": feat.get("vol_ma3"),
                "vol_rate": feat.get("vol_rate"),
                "reason": reason
            }
        return None
