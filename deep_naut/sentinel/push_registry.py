"""PUSH registration manager for kabu Station API."""

import json
import os
import threading
import time
from typing import Dict, Tuple

import requests

from .rate_limiter import RateLimiter


class PushRegistry:
    """
    Manage kabu Station PUSH registrations with TTL and priority trimming.
    """

    def __init__(self, config_path: str, max_symbols: int = 40, ttl_sec: int = 60):
        self.config_path = config_path
        self.max_symbols = max_symbols
        self.ttl_sec = ttl_sec
        self.lock = threading.Lock()
        self.symbols: Dict[str, Dict[str, float]] = {}
        self.limiter = RateLimiter(100)
        self._stop = False
        self.worker = threading.Thread(target=self._gc_loop, daemon=True, name="PushRegistryGC")
        self.worker.start()

    def _load_api(self) -> Tuple[str, str]:
        with open(self.config_path, "r", encoding="utf-8") as cfg_file:
            cfg = json.load(cfg_file)
        base = cfg["api"]["base_url"]
        token_path = os.path.join(os.path.dirname(self.config_path), "kabu_token.json")
        with open(token_path, "r", encoding="utf-8") as token_file:
            token_data = json.load(token_file)
        token = token_data.get("Token") or token_data.get("token")
        if not token:
            raise ValueError("API token missing in kabu_token.json")
        return base.rstrip("/"), token

    def refresh(self, symbol: str, priority: int = 1):
        """Record/refresh a monitoring request."""
        now = time.time()
        with self.lock:
            self.symbols[symbol] = {"last": now, "priority": priority}
        return {"ok": True, "registered": len(self.symbols)}

    def _gc_loop(self):
        while not self._stop:
            now = time.time()
            expired = []
            with self.lock:
                for sym, meta in list(self.symbols.items()):
                    if now - meta["last"] > self.ttl_sec:
                        expired.append(sym)
                for sym in expired:
                    del self.symbols[sym]
            for sym in expired:
                self._unregister_symbol(sym)

            with self.lock:
                if len(self.symbols) > self.max_symbols:
                    sorted_syms = sorted(
                        self.symbols.items(), key=lambda item: item[1]["priority"]
                    )
                    to_remove = sorted_syms[:-self.max_symbols]
                    for sym, _ in to_remove:
                        del self.symbols[sym]
                        self._unregister_symbol(sym)
            time.sleep(5)

    def _unregister_symbol(self, symbol: str):
        try:
            base, token = self._load_api()
            self.limiter.acquire()
            url = f"{base}/unregister"
            headers = {"X-API-KEY": token}
            payload = {"Symbols": [symbol]}
            requests.put(url, headers=headers, json=payload, timeout=2)
            print(f"[PUSH] Unregister {symbol}")
        except Exception as exc:
            print(f"[PUSH] Unregister failed {symbol}: {exc}")

    def ensure_registered(self, symbol: str):
        try:
            base, token = self._load_api()
            self.limiter.acquire()
            url = f"{base}/register"
            headers = {"X-API-KEY": token}
            payload = {"Symbols": [symbol]}
            requests.put(url, headers=headers, json=payload, timeout=2)
            print(f"[PUSH] Register {symbol}")
        except Exception as exc:
            print(f"[PUSH] Register failed {symbol}: {exc}")

    def stop(self):
        self._stop = True
