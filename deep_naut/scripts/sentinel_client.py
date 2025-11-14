import requests


class SentinelClient:
    def __init__(self, base_url="http://127.0.0.1:58900", timeout=2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_board(self, symbol, since=None, limit=1):
        params = {"symbol": str(symbol), "limit": int(limit)}
        if since is not None:
            params["since"] = float(since)

        url = f"{self.base_url}/api/feed/board"
        print(f"[SentinelClient] GET {url} params={params}")

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            print(f"[SentinelClient] status={resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[SentinelClient] ERROR: {e}")
            return None
