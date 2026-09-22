import pytest
from fastapi.testclient import TestClient

from app.db import (
    embedding_column_dimension,
    get_sessionmaker,
    validate_embedding_dimension,
)
from app.embeddings import HashingEmbeddingProvider
from app.main import app

pytestmark = pytest.mark.integration


def test_schema_reports_the_migrated_dimension(client: TestClient) -> None:
    with get_sessionmaker()() as session:
        assert embedding_column_dimension(session) == 384


def test_matching_provider_passes_validation(client: TestClient) -> None:
    with get_sessionmaker()() as session:
        validate_embedding_dimension(session, HashingEmbeddingProvider(dimension=384))


def test_mismatched_provider_fails_validation_with_a_clear_error(
    client: TestClient,
) -> None:
    with get_sessionmaker()() as session:
        with pytest.raises(RuntimeError, match=r"999-dim .* vector\(384\)"):
            validate_embedding_dimension(
                session, HashingEmbeddingProvider(dimension=999)
            )


def test_application_startup_runs_the_validation(client: TestClient) -> None:
    # Entering the TestClient context runs the lifespan; with the migrated
    # schema and the configured provider it must start cleanly.
    with TestClient(app) as started:
        assert started.get("/health").status_code == 200
