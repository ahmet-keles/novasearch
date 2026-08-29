"""Persist the embedding-space identity of the stored index.

Dimension validation cannot distinguish a 384-dim hashing vector from a
384-dim sentence-transformer vector, so a provider switch against a
populated index would silently compare incomparable vectors. This adds
the single-row ``embedding_space`` table: all-NULL identity means
unclaimed (empty index, any provider may adopt it on first ingest);
once claimed, ingestion verifies the claim under the row lock and
startup refuses a provider from a different space.

The table starts unclaimed, so any chunks are cleared to keep the
invariant "unclaimed space == no indexed embeddings" (a no-op directly
after 0003, which already cleared them; documents are kept and can be
re-ingested). The cache epoch is bumped so no cached search response
can outlive its deleted chunks.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-29

"""

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embedding_space",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("model_name", sa.Text(), nullable=True),
        sa.Column("dimension", sa.Integer(), nullable=True),
        sa.CheckConstraint("id = 1", name="embedding_space_single_row"),
        sa.CheckConstraint(
            "(provider IS NULL) = (dimension IS NULL)",
            name="embedding_space_claim_consistent",
        ),
    )
    op.execute(
        "INSERT INTO embedding_space (id, provider, model_name, dimension) "
        "VALUES (1, NULL, NULL, NULL)"
    )

    op.execute("DELETE FROM chunks")
    op.execute("UPDATE cache_epoch SET epoch = epoch + 1")


def downgrade() -> None:
    # Dropping the guard leaves existing chunks in whatever space produced
    # them; older application versions never read this table.
    op.drop_table("embedding_space")
