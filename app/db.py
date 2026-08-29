from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.embeddings import EmbeddingProvider


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, closed afterwards."""
    with get_sessionmaker()() as session:
        yield session


def embedding_column_dimension(session: Session) -> int:
    """The vector dimension the chunks.embedding column actually has.

    pgvector stores the dimension in the column's type modifier, so this
    reads the live schema — not the migration source or configuration.
    """
    return session.execute(
        text(
            """
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = 'chunks' AND a.attname = 'embedding'
            """
        )
    ).scalar_one()


def validate_embedding_dimension(session: Session, provider: EmbeddingProvider) -> None:
    """Fail fast when the provider and the pgvector schema disagree.

    Runs at application startup: a mismatch would otherwise surface as a
    confusing pgvector error on the first ingest, or — worse, with a
    hypothetical dimension-coercing provider — as silently incomparable
    vectors. Embeddings are provider-specific; changing the embedding
    space requires a migration and re-ingestion (see migration 0003).
    """
    schema_dimension = embedding_column_dimension(session)

    if provider.dimension != schema_dimension:
        raise RuntimeError(
            f"embedding provider produces {provider.dimension}-dim vectors but "
            f"chunks.embedding is vector({schema_dimension}); align "
            "NOVA_EMBEDDING_PROVIDER/NOVA_EMBEDDING_DIM with the schema "
            "(changing the embedding space requires a migration and re-ingestion)"
        )
