"""Semantic retrieval quality over the paraphrase fixtures.

The fixtures pair each query with a document that answers it in entirely
different words. The model provider must retrieve the target; the lexical
hashing baseline — with no vocabulary overlap to work on — must not, which
is exactly the gap that justifies model-backed embeddings.

Model-marked tests need the ``[model]`` extra and a one-time model
download; without them they skip, unless NOVA_MODEL_TESTS_REQUIRED=1
(set in the CI model job) turns unavailability into a failure.
"""

import importlib.util
import json
import os
import pathlib

import pytest
from sqlalchemy import text

from app.config import get_settings
from app.db import get_sessionmaker
from app.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from app.ingestion import ingest_document
from app.search import semantic_search

pytestmark = pytest.mark.integration

FIXTURES = json.loads(
    (pathlib.Path(__file__).parent.parent / "fixtures" / "semantic_eval.json").read_text()
)

MODEL_REQUIRED = os.environ.get("NOVA_MODEL_TESTS_REQUIRED") == "1"


@pytest.fixture(scope="session")
def model_provider() -> SentenceTransformerEmbeddingProvider:
    if importlib.util.find_spec("fastembed") is None:
        message = 'fastembed not installed (pip install -e ".[model]")'
        if MODEL_REQUIRED:
            pytest.fail(message)
        pytest.skip(message)

    settings = get_settings()
    try:
        return SentenceTransformerEmbeddingProvider(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            cache_dir=settings.embedding_cache_dir,
        )
    except Exception as error:
        if MODEL_REQUIRED:
            raise
        pytest.skip(f"embedding model unavailable: {error}")


def evaluate(provider: EmbeddingProvider) -> tuple[float, float]:
    """Ingest the fixture corpus with `provider`; return (top1, recall@3)."""
    with get_sessionmaker()() as session:
        session.execute(text("TRUNCATE documents CASCADE"))
        # Unclaim the embedding space (cleared-index invariant), so each
        # evaluation run adopts its own provider — the hashing-contrast
        # test must work even when the suite runs under the model config.
        session.execute(
            text(
                "UPDATE embedding_space"
                " SET provider = NULL, model_name = NULL, dimension = NULL"
            )
        )
        for document in FIXTURES["corpus"]:
            ingest_document(
                session,
                provider,
                title=document["title"],
                content=document["content"],
                metadata={},
            )
        session.commit()

    top1_hits = 0
    recall3_hits = 0

    with get_sessionmaker()() as session:
        for case in FIXTURES["queries"]:
            hits = semantic_search(session, provider, case["query"], limit=3)
            titles = [hit.document_title for hit in hits]

            if titles and titles[0] == case["expected_title"]:
                top1_hits += 1
            if case["expected_title"] in titles:
                recall3_hits += 1

    total = len(FIXTURES["queries"])
    return top1_hits / total, recall3_hits / total


@pytest.mark.model
def test_model_provider_retrieves_paraphrased_answers(
    migrated_database: None, model_provider: SentenceTransformerEmbeddingProvider
) -> None:
    top1, recall3 = evaluate(model_provider)

    assert recall3 == 1.0, "every paraphrase target must appear in the top 3"
    assert top1 >= 0.75, f"model top-1 accuracy {top1} below threshold"


def test_paraphrase_queries_defeat_the_hashing_baseline(
    migrated_database: None,
) -> None:
    """Documents the gap that motivates the model provider.

    The queries share no meaningful vocabulary with their targets, so the
    lexical baseline cannot reliably rank them first — if it ever could,
    the fixtures would no longer measure semantic understanding.
    """
    top1, _ = evaluate(HashingEmbeddingProvider(dimension=get_settings().embedding_dim))

    assert top1 <= 0.5, (
        f"hashing top-1 accuracy {top1}: fixtures must not be solvable lexically"
    )


@pytest.mark.model
def test_model_provider_dimension_matches_the_schema_setting(
    model_provider: SentenceTransformerEmbeddingProvider,
) -> None:
    assert model_provider.dimension == get_settings().embedding_dim
