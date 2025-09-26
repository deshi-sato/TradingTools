import logging
import time
from typing import Dict, List, Tuple, Optional


logger = logging.getLogger(__name__)
_LAST_REST_TS: Dict[str, float] = {}


class BoardFetcher:
    """
    板取得の抽象化レイヤ。
    - mode="push": kabuステ PUSH購読（未実装フック）
    - mode="rest": RESTポーリング
    - mode="auto": PUSH優先、失敗時REST
    ※ 最小構成ではダミーRESTを返す。PUSH接続は次チャットで実装。
    """

    def __init__(self, mode: str = "auto", rest_poll_ms: int = 500) -> None:
        self.mode = mode
        self.rest_poll_ms = rest_poll_ms
        
    # 実運用：ここにPUSH購読セットアップを組み込む
    def start_push(self) -> None:
        pass

    def get_board(self, symbol: str) -> Dict:
        """
        返却フォーマット:
        {
          "bid1": float|None, "ask1": float|None,
          "bids": List[(price, qty)], "asks": List[(price, qty)]
        }
        """
        now = time.monotonic() * 1000
        last_ts = _LAST_REST_TS.get(symbol, 0.0)
        logger.info("[THROTTLE?] symbol=%s last_rest_ts=%s now=%s interval=%s", symbol, last_ts, now, self.rest_poll_ms)
        if self.mode in ("auto", "rest"):
            if now - last_ts >= self.rest_poll_ms:
                # --- ダミー：実際はRESTで板取得 ---
                bid1 = 1000.0
                ask1 = 1000.5
                bids: List[Tuple[float, int]] = [
                    (1000.0, 1200),
                    (999.9, 900),
                    (999.8, 700),
                ]
                asks: List[Tuple[float, int]] = [
                    (1000.5, 1100),
                    (1000.6, 800),
                    (1000.7, 600),
                ]
                _LAST_REST_TS[symbol] = now
                logger.info("[REST] symbol=%s fetch=board", symbol)
                return {"bid1": bid1, "ask1": ask1, "bids": bids, "asks": asks}
            logger.info("[FALLBACK] symbol=%s reason=REST_GUARD", symbol)
        # フォールバック
        return {"bid1": None, "ask1": None, "bids": [], "asks": []}
