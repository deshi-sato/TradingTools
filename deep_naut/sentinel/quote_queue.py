"""Rate limited quote queue with basic coalescing."""

import threading
import time
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List, Optional, Any

from .rate_limiter import RateLimiter


Resolver = Callable[[dict], None]


@dataclass(order=True)
class QItem:
    priority: int
    ts: float
    seq: int = field(compare=False)
    kind: str = field(compare=False)
    symbol: str = field(compare=False)


class QuoteQueue:
    """
    Queue that executes quote fetches under a rate limit and coalesces duplicate
    (kind, symbol) requests so the latest invocation wins but all waiters receive
    the resulting payload.
    """

    def __init__(self, min_interval_ms: int = 100):
        self.q = PriorityQueue()
        self.limiter = RateLimiter(min_interval_ms)
        self.coalesce: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.stats = {"enq": 0, "deq": 0, "dropped": 0}
        self._id_seq = 0
        self._lock = threading.Lock()
        self._stop = False
        self.fetch_quote_fn: Optional[Callable[[str], dict]] = None
        self.worker = threading.Thread(
            target=self._run, daemon=True, name="QuoteQueueWorker"
        )
        self.worker.start()

    def set_fetcher(self, fn: Callable[[str], dict]):
        """Register the callback that performs the actual quote retrieval."""
        self.fetch_quote_fn = fn

    def stop(self):
        """Request the worker to stop."""
        self._stop = True
        # Push a sentinel to unblock the queue.
        self.q.put(QItem(0, time.monotonic(), -1, "__stop__", "__stop__"))

    def enqueue(self, kind: str, symbol: str, priority: int, resolve: Resolver):
        """Add a new request to the queue with simple coalescing."""
        with self._lock:
            self._id_seq += 1
            seq = self._id_seq
            key = (kind, symbol)
            bucket = self.coalesce.get(key)
            if bucket:
                bucket["seq"] = seq
                bucket["resolvers"].append(resolve)
                self.stats["dropped"] += 1
            else:
                bucket = {"seq": seq, "resolvers": [resolve]}
                self.coalesce[key] = bucket
            self.stats["enq"] += 1
            self.q.put(QItem(priority, time.monotonic(), seq, kind, symbol))

    def get_stats(self) -> Dict[str, int]:
        """Return a snapshot of the queue counters."""
        with self._lock:
            return dict(self.stats)

    def _run(self):
        while True:
            item: QItem = self.q.get()
            if self._stop and item.kind == "__stop__":
                self.q.task_done()
                break
            key = (item.kind, item.symbol)
            with self._lock:
                bucket = self.coalesce.get(key)
                if not bucket or bucket["seq"] != item.seq:
                    self.q.task_done()
                    continue
                resolvers: List[Resolver] = bucket["resolvers"]
                del self.coalesce[key]
            self.limiter.acquire()
            payload = self._fetch(item.symbol)
            self._resolve_all(resolvers, payload)
            with self._lock:
                self.stats["deq"] += len(resolvers)
            self.q.task_done()

    def _fetch(self, symbol: str) -> dict:
        if not self.fetch_quote_fn:
            return {"symbol": symbol, "ts": None, "price": None, "volume": None}
        try:
            return self.fetch_quote_fn(symbol)
        except Exception as exc:  # pragma: no cover - defensive logging path
            return {"error": str(exc)}

    @staticmethod
    def _resolve_all(resolvers: List[Resolver], payload: dict):
        for idx, resolver in enumerate(resolvers):
            data = payload if idx == 0 else dict(payload)
            try:
                resolver(data)
            except Exception:
                # Resolver belongs to caller context; ignore failures so the
                # queue keeps draining.
                pass
