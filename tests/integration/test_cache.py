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
