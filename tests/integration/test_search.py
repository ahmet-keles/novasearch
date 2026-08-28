import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


REDIS_DOC = {
    "title": "Redis caching guide",
    "content": (
        "Redis is an in-memory data store often used as a cache. "
        "Caching search responses in Redis reduces database load. "
        "Expiration policies keep cached entries fresh."
    ),
}

POSTGRES_DOC = {
    "title": "PostgreSQL indexing",
    "content": (
        "PostgreSQL supports GIN indexes for full-text search. "
        "The pgvector extension adds vector similarity with HNSW indexes. "
        "Combining both enables hybrid retrieval."
    ),
}

COOKING_DOC = {
    "title": "Tomato soup recipe",
    "content": (
        "Simmer ripe tomatoes with basil, garlic, and olive oil. "
        "Blend until smooth and season the soup with salt and pepper."
    ),
}


def ingest_corpus(client: TestClient) -> None:
    for doc in (REDIS_DOC, POSTGRES_DOC, COOKING_DOC):
        assert client.post("/documents", json=doc).status_code == 201


def titles(response) -> list[str]:
    return [r["document_title"] for r in response.json()["results"]]


def test_semantic_search_ranks_the_relevant_document_first(client: TestClient) -> None:
    ingest_corpus(client)

    response = client.get("/search", params={"q": "redis cache expiration", "mode": "semantic"})

    assert response.status_code == 200
    assert titles(response)[0] == "Redis caching guide"
    scores = [r["score"] for r in response.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_keyword_search_matches_full_text(client: TestClient) -> None:
    ingest_corpus(client)

    response = client.get("/search", params={"q": "pgvector HNSW", "mode": "keyword"})

    assert response.status_code == 200
    assert titles(response) == ["PostgreSQL indexing"]


def test_keyword_search_without_matches_returns_empty(client: TestClient) -> None:
    ingest_corpus(client)

    response = client.get("/search", params={"q": "zeppelin", "mode": "keyword"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_hybrid_ranks_document_found_by_both_retrievers_first(client: TestClient) -> None:
    ingest_corpus(client)

    response = client.get("/search", params={"q": "postgresql vector indexes", "mode": "hybrid"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert titles(response)[0] == "PostgreSQL indexing"


def test_hybrid_is_the_default_mode(client: TestClient) -> None:
    ingest_corpus(client)

    response = client.get("/search", params={"q": "tomato soup"})

    assert response.json()["mode"] == "hybrid"
    assert titles(response)[0] == "Tomato soup recipe"


def test_limit_bounds_the_result_count(client: TestClient) -> None:
    ingest_corpus(client)

    response = client.get("/search", params={"q": "the", "mode": "semantic", "limit": 2})

    assert len(response.json()["results"]) <= 2


def test_invalid_mode_and_empty_query_are_rejected(client: TestClient) -> None:
    assert client.get("/search", params={"q": "x", "mode": "regex"}).status_code == 422
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "limit": 0}).status_code == 422


@pytest.mark.parametrize("mode", ["semantic", "keyword", "hybrid"])
@pytest.mark.parametrize("q", ["!!!", "---", "   "])
def test_query_without_indexable_tokens_is_rejected_in_every_mode(
    client: TestClient, mode: str, q: str
) -> None:
    response = client.get("/search", params={"q": q, "mode": mode})

    assert response.status_code == 422
    assert "no indexable tokens" in response.json()["detail"]
