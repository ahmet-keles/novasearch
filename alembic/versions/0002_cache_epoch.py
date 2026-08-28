"""Move the search-cache invalidation epoch into PostgreSQL.

A single-row table holds the epoch that search cache keys embed.
Ingestion increments it in the same transaction as the document write,
so invalidation commits atomically with the data and survives any Redis
outage — Redis only ever stores disposable response payloads.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-28

"""

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cache_epoch",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("id = 1", name="cache_epoch_single_row"),
    )
    op.execute("INSERT INTO cache_epoch (id, epoch) VALUES (1, 0)")


def downgrade() -> None:
    op.drop_table("cache_epoch")
