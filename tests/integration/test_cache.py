import pytest
import redis
from fastapi.testclient import TestClient

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


def test_late_cache_write_after_invalidation_lands_in_retired_namespace(
    client: TestClient, redis_client: redis.Redis
) -> None:
    """The miss-then-write race, driven directly against SearchCache.

    A search captures version N and misses; an ingestion invalidates
    (N -> N+1) while the search is still computing; the search's late
    write must land under N and be unreachable under N+1.
    """
    from app.cache import get_search_cache

    cache = get_search_cache()

    version = cache.current_version()
    assert cache.get(version=version, mode="hybrid", query="q", limit=10) is None

    cache.invalidate_all()  # concurrent ingestion commits and invalidates

    cache.put(
        version=version, mode="hybrid", query="q", limit=10, payload={"stale": True}
    )

    fresh_version = cache.current_version()
    assert fresh_version != version
    assert cache.get(version=fresh_version, mode="hybrid", query="q", limit=10) is None, (
        "a write under the retired namespace must be invisible to new searches"
    )
    # The stale entry exists, but only under the retired namespace.
    assert cache.get(version=version, mode="hybrid", query="q", limit=10) == {"stale": True}


def test_search_racing_an_ingest_cannot_cache_stale_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end race: ingestion lands between cache miss and cache write.

    The retriever is wrapped so that, after it computes its (about to be
    stale) results, a concurrent ingestion commits and invalidates the
    cache — exactly the window in which the response is then written. The
    next search must see the new document, not the raced cache entry.
    """
    import app.routes.search as search_route
    from app.cache import get_search_cache
    from app.db import get_sessionmaker
    from app.embeddings import get_embedding_provider
    from app.ingestion import ingest_document

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
        get_search_cache().invalidate_all()

        return hits

    monkeypatch.setattr(search_route, "hybrid_search", hybrid_with_concurrent_ingest)

    raced = client.get("/search", params={"q": "kafka partition rebalancing"})
    raced_titles = [r["document_title"] for r in raced.json()["results"]]
    assert "Kafka partition rebalancing" not in raced_titles, (
        "the raced request itself predates the ingest and serves old results"
    )

    monkeypatch.setattr(search_route, "hybrid_search", real_hybrid)

    after = client.get("/search", params={"q": "kafka partition rebalancing"})
    titles = [r["document_title"] for r in after.json()["results"]]

    assert titles[0] == "Kafka partition rebalancing", (
        "the raced response must not have been cached under the new namespace"
    )


def test_ingestion_invalidates_cached_search_responses(client: TestClient) -> None:
    client.post("/documents", json=DOC)

    before = client.get("/search", params={"q": "kafka partition rebalancing"})
    assert [r["document_title"] for r in before.json()["results"]] == ["Kafka partitioning"]

    client.post("/documents", json=BETTER_DOC)

    after = client.get("/search", params={"q": "kafka partition rebalancing"})
    titles = [r["document_title"] for r in after.json()["results"]]

    assert titles[0] == "Kafka partition rebalancing", (
        "a search after ingestion must see the new document, not a stale cache entry"
    )
