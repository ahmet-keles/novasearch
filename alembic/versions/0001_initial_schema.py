"""Initial schema: documents, chunks, search indexes.

The embedding dimension (256) is fixed here and must match
NOVA_EMBEDDING_DIM (app.config.Settings.embedding_dim); a mismatch fails
at insert time rather than corrupting data.

Revision ID: 0001
Revises:
Create Date: 2026-08-28

"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, TSVECTOR, UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 256


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "tsv",
            TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
        ),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )

    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # Keyword search: GIN over the generated tsvector.
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")

    # Semantic search: HNSW over cosine distance, matching the operator the
    # query path uses (embedding <=> :query).
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
