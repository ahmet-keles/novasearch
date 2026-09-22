"""Widen chunk embeddings to 384 dimensions.

384 is the output dimension of sentence-transformers/all-MiniLM-L6-v2,
the default model of the model-backed embedding provider; the hashing
provider is dimension-parametric and follows along. One deployment has
one embedding space: vectors from different providers or dimensions are
not comparable, so existing chunk rows — derived data, regenerable by
re-ingesting the verbatim documents — are deleted rather than pretending
a 256-dim vector can become a 384-dim one. The cache epoch is bumped in
the same migration so no cached search response referencing the deleted
chunks survives.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-29

"""

from pgvector.sqlalchemy import Vector

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("DELETE FROM chunks")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.alter_column("chunks", "embedding", type_=Vector(EMBEDDING_DIM))
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("UPDATE cache_epoch SET epoch = epoch + 1")


def downgrade() -> None:
    op.execute("DELETE FROM chunks")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.alter_column("chunks", "embedding", type_=Vector(256))
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("UPDATE cache_epoch SET epoch = epoch + 1")
