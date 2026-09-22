import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class CacheEpoch(Base):
    """Single-row table owning the search-cache invalidation epoch.

    Ingestion increments the epoch in the same transaction as the document
    write, and search cache keys embed it. PostgreSQL — not Redis — is the
    source of truth for invalidation, so a Redis outage can never lose an
    epoch bump and recovered Redis entries under old epochs stay
    unreachable. The CHECK pins the table to one row; the UPDATE briefly
    serializes concurrent ingestions on its row lock.
    """

    __tablename__ = "cache_epoch"
    __table_args__ = (CheckConstraint("id = 1", name="cache_epoch_single_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)


class EmbeddingSpaceRecord(Base):
    """Single-row table owning the identity of the stored embedding space.

    All-NULL identity means unclaimed: no chunks are indexed and the next
    ingestion may adopt whichever provider is configured. Once claimed,
    ingestion verifies (under the row lock) that its provider produces the
    same space, and startup refuses a provider whose space differs — equal
    dimensions alone never make vectors from two providers comparable.
    The CHECKs pin the table to one row and keep the identity columns
    NULL or set together.
    """

    __tablename__ = "embedding_space"
    __table_args__ = (
        CheckConstraint("id = 1", name="embedding_space_single_row"),
        CheckConstraint(
            "(provider IS NULL) = (dimension IS NULL)",
            name="embedding_space_claim_consistent",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Document(Base):
    """An ingested document, kept verbatim; search runs over its chunks."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
    )


class Chunk(Base):
    """One retrieval unit of a document.

    tsv is a stored generated column, so the keyword index can never drift
    from the content; embedding is written by the application at ingest time.
    """

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dim), nullable=False
    )
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
