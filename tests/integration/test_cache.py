import pytest
import redis
from fastapi.testclient import TestClient

from app.cache import bump_epoch, current_epoch, get_search_cache
from app.db import get_sessionmaker
from app.embeddings import get_embedding_provider
from app.ingestion import ingest_document

pytestmark = pytest.mark.integration


DOC = {
    "title": "Kafka partitioning",
    "content": "Kafka topics are split into partitions; keys route records to partitions.",
}

BETTER_DOC = {
    "title": "Kafka partition rebalancing",
    "content": (
        "Kafka partition assignment and partition rebalancing move partitions "
        "between consumers. Partition ownership changes during a rebalance."
    ),
}

QUERY = {"q": "kafka partition rebalancing"}


class _DownRedis:
    """Stands in for SearchCache's client while Redis is 'down'."""

    def __getattr__(self, name: str):
        def _raise(*args, **kwargs):
            raise redis.ConnectionError("redis is down (simulated)")

        return _raise


def titles(response) -> list[str]:
    return [r["document_title"] for r in response.json()["results"]]


def test_search_responses_are_cached_in_redis(
    client: TestClient, redis_client: redis.Redis
) -> None:
    client.post("/documents", json=DOC)

    first = client.get("/search", params={"q": "kafka partitions"})
    second = client.get("/search", params={"q": "kafka partitions"})

    assert first.json() == second.json()
    assert redis_client.keys("novasearch:search:entry:*"), (
        "a cache entry must exist after a search"
    )


def test_late_cache_write_after_epoch_bump_lands_in_retired_epoch(
    client: TestClient,
) -> None:
    """The miss-then-write race, driven directly against the cache layer.

    A search captures epoch N and misses; an ingestion commits an epoch
    bump (N -> N+1) while the search is still computing; the search's late
    write must land under N and be unreachable under N+1.
    """
    cache = get_search_cache()

    with get_sessionmaker()() as session:
        epoch = current_epoch(session)

    assert cache.get(epoch=epoch, mode="hybrid", query="q", limit=10) is None

    with get_sessionmaker()() as session:  # concurrent ingestion commits
        bump_epoch(session)
        session.commit()

    cache.put(epoch=epoch, mode="hybrid", query="q", limit=10, payload={"stale": True})

    with get_sessionmaker()() as session:
        fresh_epoch = current_epoch(session)

    assert fresh_epoch != epoch
    assert cache.get(epoch=fresh_epoch, mode="hybrid", query="q", limit=10) is None, (
        "a write under the retired epoch must be invisible to new searches"
    )
    # The stale entry exists, but only under the retired epoch.
    assert cache.get(epoch=epoch, mode="hybrid", query="q", limit=10) == {"stale": True}


def test_search_racing_an_ingest_cannot_cache_stale_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end race: ingestion lands between cache miss and cache write.

    The retriever is wrapped so that, after it computes its (about to be
    stale) results, a concurrent ingestion commits — bumping the epoch in
    its own transaction — exactly in the window before the response is
    written to the cache. The next search must see the new document.
    """
    import app.routes.search as search_route

    client.post("/documents", json=DOC)

    real_hybrid = search_route.hybrid_search

    def hybrid_with_concurrent_ingest(session, provider, query, limit):
        hits = real_hybrid(session, provider, query, limit)

        with get_sessionmaker()() as ingest_session:
            ingest_document(
                ingest_session,
                get_embedding_provider(),
                title=BETTER_DOC["title"],
                content=BETTER_DOC["content"],
                metadata={},
            )
            ingest_session.commit()

        return hits

    monkeypatch.setattr(search_route, "hybrid_search", hybrid_with_concurrent_ingest)

    raced = client.get("/search", params=QUERY)
    assert "Kafka partition rebalancing" not in titles(raced), (
        "the raced request itself predates the ingest and serves old results"
    )

    monkeypatch.setattr(search_route, "hybrid_search", real_hybrid)

    after = client.get("/search", params=QUERY)

    assert titles(after)[0] == "Kafka partition rebalancing", (
        "the raced response must not have been cached under the new epoch"
    )


def test_redis_outage_during_ingestion_cannot_resurrect_stale_entries(
    client: TestClient, redis_client: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalidation survives a Redis outage because PostgreSQL owns it.

    Scenario: a response is cached at epoch N; Redis goes down; an
    ingestion still commits (bumping the epoch in PostgreSQL); Redis comes
    back before the old entry's TTL expires. The epoch-N entry is still
    physically in Redis — and must never be served again.
    """
    cache = get_search_cache()

    client.post("/documents", json=DOC)
    before = client.get("/search", params=QUERY)
    assert titles(before) == ["Kafka partitioning"]

    stale_keys = redis_client.keys("novasearch:search:entry:*")
    assert stale_keys, "the pre-outage response must be cached"

    real_client = cache._client
    monkeypatch.setattr(cache, "_client", _DownRedis())  # Redis goes down

    assert client.get("/health").status_code == 503

    created = client.post("/documents", json=BETTER_DOC)
    assert created.status_code == 201, "ingestion must succeed while Redis is down"

    during = client.get("/search", params=QUERY)
    assert titles(during)[0] == "Kafka partition rebalancing", (
        "search during the outage is uncached but must be correct"
    )

    monkeypatch.setattr(cache, "_client", real_client)  # Redis comes back

    assert all(redis_client.exists(key) for key in stale_keys), (
        "the stale epoch-N entry is still physically present in Redis"
    )

    after = client.get("/search", params=QUERY)
    assert titles(after)[0] == "Kafka partition rebalancing", (
        "the recovered Redis must not serve the stale epoch-N entry: the "
        "PostgreSQL epoch moved on with the ingestion"
    )


def test_ingestion_invalidates_cached_search_responses(client: TestClient) -> None:
    client.post("/documents", json=DOC)

    before = client.get("/search", params=QUERY)
    assert titles(before) == ["Kafka partitioning"]

    client.post("/documents", json=BETTER_DOC)

    after = client.get("/search", params=QUERY)

    assert titles(after)[0] == "Kafka partition rebalancing", (
        "a search after ingestion must see the new document, not a stale cache entry"
    )
