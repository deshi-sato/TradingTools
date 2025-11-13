"""HTTPクライアントでセンチネルAPIとやり取りする薄いラッパー."""

from __future__ import annotations

import requests


class SentinelClient:
    """センチネルサーバーへの最小限の通信をまとめたクライアント."""

    def __init__(self, base_url: str = "http://127.0.0.1:58900", timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_board(self, symbol, since=None, limit: int = 1):
        """板情報を取得する."""
        params = {"symbol": str(symbol), "limit": int(limit)}
        if since is not None:
            params["since"] = float(since)
        response = requests.get(
            f"{self.base_url}/api/feed/board",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def paper_order(
        self,
        symbol,
        side: str,
        qty,
        order_type: str,
        price=None,
        tif: str = "IOC",
    ):
        """仮想発注APIを呼び出す."""
        payload = {
            "symbol": str(symbol),
            "side": side,
            "qty": int(qty),
            "type": order_type,
            "tif": tif,
        }
        if price is not None:
            payload["price"] = float(price)
        response = requests.post(
            f"{self.base_url}/api/order/paper",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
