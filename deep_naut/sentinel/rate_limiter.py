"""Simple monotonic clock based rate limiter."""

import time


class RateLimiter:
    """Enforces a minimum delay between successive acquire calls."""

    def __init__(self, min_interval_ms: int = 100):
        self.min_interval = max(1, min_interval_ms) / 1000.0
        self._next_at = 0.0

    def acquire(self):
        """Sleep until the minimum interval has passed."""
        now = time.monotonic()
        wait = self._next_at - now
        if wait > 0:
            time.sleep(wait)
        self._next_at = time.monotonic() + self.min_interval
