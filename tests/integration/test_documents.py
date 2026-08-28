import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import get_engine

pytestmark = pytest.mark.integration


def test_ingestion_persists_document_chunks_and_embeddings(client: TestClient) -> None:
    content = "PostgreSQL full-text search pairs well with vector similarity. " * 60

    response = client.post(
        "/documents",
        json={"title": "Hybrid search", "content": content, "metadata": {"source": "test"}},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["chunk_count"] > 1

    with get_engine().connect() as connection:
        chunk_count, tsv_count, embedding_dim = connection.execute(
            text(
                """
                SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE tsv IS NOT NULL),
                       MAX(vector_dims(embedding))
                FROM chunks
                WHERE document_id = :id
                """
            ),
            {"id": body["id"]},
        ).one()

    assert chunk_count == body["chunk_count"]
    assert tsv_count == chunk_count, "generated tsvector must be populated for every chunk"
    assert embedding_dim == 256


def test_document_can_be_read_back(client: TestClient) -> None:
    created = client.post(
        "/documents",
        json={"title": "Read me", "content": "A short document about Redis caching."},
    ).json()

    response = client.get(f"/documents/{created['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Read me"
    assert body["chunk_count"] == created["chunk_count"]
    assert body["metadata"] == {}


def test_unknown_document_returns_404(client: TestClient) -> None:
    response = client.get("/documents/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404


def test_content_without_indexable_tokens_is_rejected(client: TestClient) -> None:
    response = client.post("/documents", json={"title": "Empty", "content": "   \n\t "})

    assert response.status_code == 422


def test_missing_fields_are_rejected(client: TestClient) -> None:
    response = client.post("/documents", json={"title": "No content"})

    assert response.status_code == 422
