import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.embeddings import (
    HashingEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_embedding_provider,
)


def test_default_provider_is_hashing_at_the_configured_dimension() -> None:
    provider = get_embedding_provider()

    assert isinstance(provider, HashingEmbeddingProvider)
    assert provider.dimension == get_settings().embedding_dim


def test_unknown_provider_name_is_rejected_at_configuration_time() -> None:
    with pytest.raises(ValidationError):
        Settings(embedding_provider="telepathy")


def test_model_provider_rejects_non_positive_batch_size() -> None:
    # Validated before any model import or download is attempted.
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddingProvider("any-model", batch_size=0)
