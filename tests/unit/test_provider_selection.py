import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.embeddings import (
    HashingEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_embedding_provider,
)


def test_default_provider_is_hashing_at_the_configured_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This asserts the *default* selection, so the ambient environment must
    # not leak in: the CI model job exports NOVA_EMBEDDING_PROVIDER=model
    # for the whole suite. Pin the variable away and rebuild the cached
    # settings/provider around the assertion, leaving the caches empty so
    # later tests re-derive them from the real environment.
    monkeypatch.delenv("NOVA_EMBEDDING_PROVIDER", raising=False)
    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    try:
        provider = get_embedding_provider()

        assert isinstance(provider, HashingEmbeddingProvider)
        assert provider.dimension == get_settings().embedding_dim
    finally:
        get_settings.cache_clear()
        get_embedding_provider.cache_clear()


def test_unknown_provider_name_is_rejected_at_configuration_time() -> None:
    with pytest.raises(ValidationError):
        Settings(embedding_provider="telepathy")


def test_model_provider_rejects_non_positive_batch_size() -> None:
    # Validated before any model import or download is attempted.
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddingProvider("any-model", batch_size=0)
