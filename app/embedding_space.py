"""Persisted embedding-space identity: one index, one embedding space.

Dimension validation alone cannot tell a 384-dim hashing vector from a
384-dim MiniLM vector, so switching NOVA_EMBEDDING_PROVIDER against an
already-populated index would silently compare vectors from unrelated
spaces. The single-row ``embedding_space`` table closes that hole:

- unclaimed (all-NULL identity) — the index holds no embeddings; any
  provider may adopt the space on its first ingest.
- claimed — every later ingest must produce the same space (verified
  under the row lock, in the same transaction as the chunk write, so
  mixing is impossible even across replicas), and startup refuses a
  provider whose space differs.

Changing the embedding space of a populated index requires a migration
that clears the chunks and resets this row — never a config flip.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingSpace
from app.models import EmbeddingSpaceRecord


class EmbeddingSpaceMismatch(RuntimeError):
    """The configured provider's space differs from the stored one."""


def _mismatch(stored: EmbeddingSpace, configured: EmbeddingSpace) -> EmbeddingSpaceMismatch:
    return EmbeddingSpaceMismatch(
        f"the index holds {stored.describe()} embeddings but the configured "
        f"provider produces {configured.describe()}; vectors from different "
        "embedding spaces are not comparable even at equal dimensions — "
        "changing the embedding space requires a migration that clears the "
        "chunks (and re-ingestion), not a provider switch"
    )


def _to_space(record: EmbeddingSpaceRecord) -> EmbeddingSpace | None:
    if record.provider is None or record.dimension is None:
        return None
    return EmbeddingSpace(
        provider=record.provider,
        model_name=record.model_name,
        dimension=record.dimension,
    )


def stored_embedding_space(session: Session) -> EmbeddingSpace | None:
    """The claimed space of the stored embeddings, or None when unclaimed."""
    record = session.execute(select(EmbeddingSpaceRecord)).scalar_one()
    return _to_space(record)


def validate_embedding_space(session: Session, space: EmbeddingSpace) -> None:
    """Fail fast when stored embeddings belong to a different space.

    Runs at application startup. An unclaimed identity passes: an empty
    index may adopt whichever provider is configured (the claim itself
    happens transactionally on the first ingest).
    """
    stored = stored_embedding_space(session)
    if stored is not None and stored != space:
        raise _mismatch(stored, space)


def claim_embedding_space(session: Session, space: EmbeddingSpace) -> None:
    """Adopt or verify the embedding space inside the caller's transaction.

    Locks the single row (FOR UPDATE) so concurrent ingestions — including
    ones on other replicas — serialize on the claim: either the space is
    unclaimed and becomes ``space`` atomically with the caller's chunk
    write, or it must already equal ``space``. A chunk can therefore never
    commit under a different space than the claim, which is what makes
    mixing structurally impossible rather than merely checked at startup.
    """
    record = session.execute(
        select(EmbeddingSpaceRecord).with_for_update()
    ).scalar_one()

    stored = _to_space(record)
    if stored is None:
        record.provider = space.provider
        record.model_name = space.model_name
        record.dimension = space.dimension
        session.flush()
        return

    if stored != space:
        raise _mismatch(stored, space)
