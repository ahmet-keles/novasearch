"""Optional cross-encoder reranking for a short hybrid candidate list."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from app.config import get_settings

if TYPE_CHECKING:
    from app.search import SearchHit


class CrossEncoder(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class FastEmbedCrossEncoder:
    """Score query-document pairs with FastEmbed's local ONNX cross-encoder."""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "cross-encoder reranking requires the [rerank] extra: "
                'pip install -e ".[rerank]"'
            ) from error

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(score) for score in self._model.rerank(query, documents)]


def rerank_hits(
    query: str, hits: list[SearchHit], reranker: CrossEncoder
) -> list[SearchHit]:
    """Sort candidate hits by cross-encoder score, retaining scores in results."""
    if not hits:
        return []

    documents = [f"{hit.document_title}\n\n{hit.content}" for hit in hits]
    scores = reranker.rerank(query, documents)
    if len(scores) != len(hits):
        raise RuntimeError("cross-encoder returned a different number of scores")

    scored = [
        replace(hit, score=score) for hit, score in zip(hits, scores, strict=True)
    ]
    return sorted(scored, key=lambda hit: (-hit.score, str(hit.chunk_id)))


@lru_cache
def get_cross_encoder_reranker() -> FastEmbedCrossEncoder:
    """Load the configured model only when enabled hybrid search needs it."""
    return FastEmbedCrossEncoder(get_settings().rerank_model)
