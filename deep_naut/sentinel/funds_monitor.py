"""Funds monitor polling kabu Station wallet and positions."""

import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, Any

import requests

from .rate_limiter import RateLimiter


class FundsMonitor:
    """
    Periodically polls kabu Station for cash/positions and exposes snapshots.
    """

    def __init__(
        self,
        config_path: str,
        poll_sec: float = 2.0,
        min_cash: float = 300000,
    ):
        self.config_path = config_path
        self.poll_sec = poll_sec
        self.min_cash = min_cash
        self.lock = threading.Lock()
        self.limiter = RateLimiter(200)
        self.snapshot: Dict[str, Any] = {
            "cash": 0,
            "positions": [],
            "pnl_today": 0,
            "updated_at": None,
            "ok": False,
        }
        self._stop = False
        self.worker = threading.Thread(target=self._loop, daemon=True, name="FundsMonitor")
        self.worker.start()

    def _load_api(self):
        with open(self.config_path, "r", encoding="utf-8") as cfg_file:
            cfg = json.load(cfg_file)
        base = cfg["api"]["base_url"].rstrip("/")
        token_path = os.path.join(os.path.dirname(self.config_path), "kabu_token.json")
        with open(token_path, "r", encoding="utf-8") as token_file:
            token_data = json.load(token_file)
        token = token_data.get("Token") or token_data.get("token")
        if not token:
            raise ValueError("API token not found in kabu_token.json")
        return base, token

    def _fetch_wallet(self, base, token):
        url = f"{base}/wallet/cash"
        headers = {"X-API-KEY": token}
        resp = requests.get(url, headers=headers, timeout=2)
        resp.raise_for_status()
        return resp.json()

    def _fetch_positions(self, base, token):
        url = f"{base}/positions"
        headers = {"X-API-KEY": token}
        resp = requests.get(url, headers=headers, timeout=2)
        resp.raise_for_status()
        return resp.json()

    def _loop(self):
        while not self._stop:
            try:
                base, token = self._load_api()
                self.limiter.acquire()
                wallet = self._fetch_wallet(base, token)
                self.limiter.acquire()
                positions = self._fetch_positions(base, token)
                cash = wallet.get("Cash", 0)
                pnl_today = sum(pos.get("ProfitLoss", 0) for pos in positions)
                with self.lock:
                    self.snapshot.update(
                        {
                            "cash": cash,
                            "positions": positions,
                            "pnl_today": pnl_today,
                            "updated_at": datetime.now().strftime("%H:%M:%S"),
                            "ok": True,
                            "error": None,
                        }
                    )
            except Exception as exc:
                with self.lock:
                    self.snapshot["ok"] = False
                    self.snapshot["error"] = str(exc)
            time.sleep(self.poll_sec)

    def get_snapshot(self):
        with self.lock:
            return dict(self.snapshot)

    def has_enough_cash(self, amount: float) -> bool:
        with self.lock:
            return self.snapshot.get("cash", 0) >= max(self.min_cash, amount)

    def stop(self):
        self._stop = True
