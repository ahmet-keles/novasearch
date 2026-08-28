import math

import pytest

from app.embeddings import HashingEmbeddingProvider


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture
def provider() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dimension=256)


def test_vectors_have_the_configured_dimension(provider: HashingEmbeddingProvider) -> None:
    [vector] = provider.embed(["hello world"])

    assert provider.dimension == 256
    assert len(vector) == 256


def test_embedding_is_deterministic(provider: HashingEmbeddingProvider) -> None:
    text = "PostgreSQL full-text search with pgvector"

    assert provider.embed([text]) == provider.embed([text])
    assert HashingEmbeddingProvider(dimension=256).embed([text]) == provider.embed([text])


def test_vectors_are_l2_normalized(provider: HashingEmbeddingProvider) -> None:
    [vector] = provider.embed(["some words to embed for the norm check"])

    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_text_without_tokens_embeds_to_zero_vector(provider: HashingEmbeddingProvider) -> None:
    [vector] = provider.embed(["!!! ... ---"])

    assert vector == [0.0] * 256


def test_shared_vocabulary_scores_higher_than_disjoint(
    provider: HashingEmbeddingProvider,
) -> None:
    base, overlapping, disjoint = provider.embed(
        [
            "postgres vector database search",
            "postgres database indexing",
            "tomato basil mozzarella salad",
        ]
    )

    assert cosine(base, overlapping) > cosine(base, disjoint)


def test_case_and_punctuation_do_not_change_the_vector(
    provider: HashingEmbeddingProvider,
) -> None:
    [a] = provider.embed(["Hybrid Search!"])
    [b] = provider.embed(["hybrid search"])

    assert a == b


def test_batch_preserves_order(provider: HashingEmbeddingProvider) -> None:
    batch = provider.embed(["first text", "second text"])
    singles = [provider.embed(["first text"])[0], provider.embed(["second text"])[0]]

    assert batch == singles


def test_invalid_dimension_is_rejected() -> None:
    with pytest.raises(ValueError):
        HashingEmbeddingProvider(dimension=0)
