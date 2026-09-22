"""The persisted embedding-space guard.

Every test here is deterministic and model-free: the model side of each
scenario is expressed through space identities (and a stub provider that
claims the model space), so the guard's behavior is pinned without a
model download. The CI model job additionally runs the whole suite with
the real provider.
"""

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_sessionmaker
from app.embedding_space import (
    EmbeddingSpaceMismatch,
    claim_embedding_space,
    stored_embedding_space,
    validate_embedding_space,
)
from app.embeddings import EmbeddingProvider, EmbeddingSpace, HashingEmbeddingProvider
from app.ingestion import ingest_document
from app.main import app
from app.models import Document

pytestmark = pytest.mark.integration


def hashing_space() -> EmbeddingSpace:
    return EmbeddingSpace(
        provider="hashing", model_name=None, dimension=get_settings().embedding_dim
    )


def model_space() -> EmbeddingSpace:
    # The identity the configured sentence-transformer provider would
    # claim — same dimension as the hashing space on purpose: dimension
    # equality alone must never make the two spaces interchangeable.
    settings = get_settings()
    return EmbeddingSpace(
        provider="model",
        model_name=settings.embedding_model,
        dimension=settings.embedding_dim,
    )


class ModelSpaceStubProvider:
    """Claims the model space while embedding hermetically (via hashing).

    Lets tests exercise "the index holds model-space embeddings" without
    downloading a model: the guard keys on the claimed space identity,
    not on vector values.
    """

    def __init__(self) -> None:
        self._inner = HashingEmbeddingProvider(dimension=get_settings().embedding_dim)

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def space(self) -> EmbeddingSpace:
        return model_space()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.embed(texts)


def ingest_sample(session: Session, provider: EmbeddingProvider) -> None:
    ingest_document(
        session,
        provider,
        title="sample",
        content="a small sample document about embedding spaces",
        metadata={},
    )


def test_hashing_data_accepts_hashing_startup(client: TestClient) -> None:
    with get_sessionmaker()() as session:
        ingest_sample(session, HashingEmbeddingProvider(dimension=384))
        session.commit()

        assert stored_embedding_space(session) == hashing_space()
        validate_embedding_space(session, hashing_space())


def test_hashing_data_rejects_model_startup_despite_equal_dimensions(
    client: TestClient,
) -> None:
    with get_sessionmaker()() as session:
        ingest_sample(session, HashingEmbeddingProvider(dimension=384))
        session.commit()

        assert hashing_space().dimension == model_space().dimension
        with pytest.raises(EmbeddingSpaceMismatch, match="not comparable"):
            validate_embedding_space(session, model_space())


def test_model_data_rejects_hashing_startup(client: TestClient) -> None:
    with get_sessionmaker()() as session:
        ingest_sample(session, ModelSpaceStubProvider())
        session.commit()

        assert stored_embedding_space(session) == model_space()
        with pytest.raises(EmbeddingSpaceMismatch, match="not comparable"):
            validate_embedding_space(session, hashing_space())


def test_empty_index_accepts_either_provider(client: TestClient) -> None:
    with get_sessionmaker()() as session:
        assert stored_embedding_space(session) is None
        validate_embedding_space(session, hashing_space())
        validate_embedding_space(session, model_space())


def test_ingestion_cannot_mix_providers(client: TestClient) -> None:
    with get_sessionmaker()() as session:
        ingest_sample(session, HashingEmbeddingProvider(dimension=384))
        session.commit()

    with get_sessionmaker()() as session:
        with pytest.raises(EmbeddingSpaceMismatch, match="not comparable"):
            ingest_sample(session, ModelSpaceStubProvider())
        session.rollback()

        # The claim happens in the same transaction as the write, before
        # anything is persisted: the rejected ingest leaves no trace and
        # the stored space is still the original one.
        assert session.execute(select(func.count(Document.id))).scalar_one() == 1
        assert stored_embedding_space(session) == hashing_space()


def test_startup_refuses_a_claimed_foreign_space(client: TestClient) -> None:
    # A space matching neither the hashing nor the configured model
    # provider, so this refusal is deterministic under both CI configs.
    foreign = EmbeddingSpace(
        provider="model",
        model_name="example/some-other-model",
        dimension=get_settings().embedding_dim,
    )
    with get_sessionmaker()() as session:
        claim_embedding_space(session, foreign)
        session.commit()

    with pytest.raises(EmbeddingSpaceMismatch, match="not comparable"):
        with TestClient(app):
            pass


def test_startup_accepts_the_unclaimed_index(client: TestClient) -> None:
    with TestClient(app) as started:
        assert started.get("/health").status_code == 200
