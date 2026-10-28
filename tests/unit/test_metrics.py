import pytest

from app.metrics import Metrics, embed_with_metrics


def test_snapshot_reports_nearest_rank_p95_and_cache_hit_rate() -> None:
    metrics = Metrics(latency_window=100)
    for latency in range(1, 101):
        metrics.observe_request(float(latency))
    metrics.record_cache_result(hit=True)
    metrics.record_cache_result(hit=False)
    metrics.record_cache_result(hit=True)

    snapshot = metrics.snapshot()

    assert snapshot["request_latency"] == {
        "sample_count": 100,
        "window_size": 100,
        "p95_ms": 95.0,
    }
    assert snapshot["search_cache"] == {
        "hits": 2,
        "misses": 1,
        "hit_rate": 0.6667,
    }


def test_embedding_call_records_count_texts_and_elapsed_time() -> None:
    metrics = Metrics()

    class Provider:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    assert embed_with_metrics(Provider(), ["one", "two"], collector=metrics) == [
        [1.0],
        [1.0],
    ]
    assert metrics.snapshot()["embeddings"]["calls"] == 1
    assert metrics.snapshot()["embeddings"]["texts"] == 2
    assert metrics.snapshot()["embeddings"]["total_time_ms"] >= 0


def test_embedding_attempt_is_counted_when_provider_raises() -> None:
    metrics = Metrics()

    class BrokenProvider:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        embed_with_metrics(BrokenProvider(), ["failed input"], collector=metrics)

    assert metrics.snapshot()["embeddings"]["calls"] == 1
    assert metrics.snapshot()["embeddings"]["texts"] == 1
