"""Retrieval: semantic (pgvector), keyword (PostgreSQL FTS), and hybrid.

Hybrid ranking uses Reciprocal Rank Fusion (RRF): each retriever
contributes a candidate ranking, and a chunk's fused score is
``sum(weight / (k + rank))`` over the rankings that contain it. RRF works
on ranks, not raw scores, so cosine similarities and ts_rank values never
need to be calibrated against each other.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.embeddings import EmbeddingProvider
from app.models import Chunk, Document


class SearchMode(StrEnum):
    semantic = "semantic"
    keyword = "keyword"
    hybrid = "hybrid"


@dataclass(frozen=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    document_title: str
    chunk_index: int
    content: str
    score: float


K = TypeVar("K", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[K]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[K, float]:
    """Fuse rankings into ``{item: score}``; ranks are 1-based."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings in length")

    scores: dict[K, float] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)

    return scores


def semantic_search(
    session: Session,
    provider: EmbeddingProvider,
    query: str,
    limit: int,
) -> list[SearchHit]:
    """Nearest chunks by cosine distance; score is cosine similarity."""
    [query_embedding] = provider.embed([query])

    # A zero vector has no direction, so cosine distance against it is
    # undefined. The API rejects tokenless queries before reaching here;
    # this guard keeps any other caller from ranking on NaN distances.
    if not any(query_embedding):
        return []

    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")

    rows = session.execute(
        select(Chunk, Document.title, distance)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(limit)
    ).all()

    return [
        _hit(chunk, title, score=1.0 - float(dist)) for chunk, title, dist in rows
    ]


def keyword_search(session: Session, query: str, limit: int) -> list[SearchHit]:
    """Full-text matches ranked by ts_rank_cd over the generated tsvector."""
    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery).label("rank")

    rows = session.execute(
        select(Chunk, Document.title, rank)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    ).all()

    return [_hit(chunk, title, score=float(r)) for chunk, title, r in rows]


def hybrid_search(
    session: Session,
    provider: EmbeddingProvider,
    query: str,
    limit: int,
) -> list[SearchHit]:
    """RRF over a candidate pool from each retriever; score is the RRF score."""
    pool = max(get_settings().search_candidate_pool, limit)

    semantic_hits = semantic_search(session, provider, query, pool)
    keyword_hits = keyword_search(session, query, pool)

    fused = reciprocal_rank_fusion(
        [
            [hit.chunk_id for hit in semantic_hits],
            [hit.chunk_id for hit in keyword_hits],
        ]
    )

    hits_by_id = {hit.chunk_id: hit for hit in semantic_hits}
    hits_by_id.update({hit.chunk_id: hit for hit in keyword_hits})

    top = sorted(fused.items(), key=lambda item: (-item[1], str(item[0])))[:limit]

    return [
        SearchHit(
            chunk_id=hits_by_id[chunk_id].chunk_id,
            document_id=hits_by_id[chunk_id].document_id,
            document_title=hits_by_id[chunk_id].document_title,
            chunk_index=hits_by_id[chunk_id].chunk_index,
            content=hits_by_id[chunk_id].content,
            score=score,
        )
        for chunk_id, score in top
    ]


def _hit(chunk: Chunk, title: str, *, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_title=title,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        score=score,
    )
