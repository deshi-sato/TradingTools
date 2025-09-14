import time
from typing import Dict, List, Tuple, Optional


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
        self._last_rest_ts = 0.0

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
        if self.mode in ("auto", "rest"):
            if now - self._last_rest_ts >= self.rest_poll_ms:
                self._last_rest_ts = now
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
                return {"bid1": bid1, "ask1": ask1, "bids": bids, "asks": asks}
        # フォールバック
        return {"bid1": None, "ask1": None, "bids": [], "asks": []}
