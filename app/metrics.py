"""Small in-process operational metrics for the NovaSearch API."""

from collections import deque
from collections.abc import Sequence
from math import ceil
from threading import Lock
from time import perf_counter


class Metrics:
    """Thread-safe counters and a bounded window of request latencies."""

    def __init__(self, *, latency_window: int = 1000) -> None:
        if latency_window <= 0:
            raise ValueError("latency_window must be positive")
        self._lock = Lock()
        self._latencies_ms: deque[float] = deque(maxlen=latency_window)
        self._cache_hits = 0
        self._cache_misses = 0
        self._embedding_calls = 0
        self._embedding_texts = 0
        self._embedding_time_ms = 0.0

    def observe_request(self, duration_ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(duration_ms)

    def record_cache_result(self, *, hit: bool) -> None:
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def record_embedding(self, *, text_count: int, duration_ms: float) -> None:
        with self._lock:
            self._embedding_calls += 1
            self._embedding_texts += text_count
            self._embedding_time_ms += duration_ms

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            hits = self._cache_hits
            misses = self._cache_misses
            calls = self._embedding_calls
            texts = self._embedding_texts
            embedding_time_ms = self._embedding_time_ms

        total_cache_lookups = hits + misses
        p95_rank = ceil(0.95 * len(latencies))
        p95 = latencies[p95_rank - 1] if latencies else None
        return {
            "request_latency": {
                "sample_count": len(latencies),
                "window_size": self._latencies_ms.maxlen,
                "p95_ms": round(p95, 3) if p95 is not None else None,
            },
            "search_cache": {
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total_cache_lookups, 4)
                if total_cache_lookups
                else None,
            },
            "embeddings": {
                "calls": calls,
                "texts": texts,
                "total_time_ms": round(embedding_time_ms, 3),
            },
        }

    def reset(self) -> None:
        """Clear collected values; intended for deterministic tests."""
        with self._lock:
            self._latencies_ms.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._embedding_calls = 0
            self._embedding_texts = 0
            self._embedding_time_ms = 0.0


metrics = Metrics()


def embed_with_metrics(
    provider, texts: Sequence[str], *, collector: Metrics | None = None
) -> list[list[float]]:
    """Run a provider call and record its count and elapsed wall time."""
    collector = collector or metrics
    started = perf_counter()
    try:
        return provider.embed(texts)
    finally:
        collector.record_embedding(
            text_count=len(texts), duration_ms=(perf_counter() - started) * 1000
        )
