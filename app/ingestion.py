from sqlalchemy.orm import Session

from app.cache import bump_epoch
from app.chunking import chunk_text
from app.config import get_settings
from app.embedding_space import claim_embedding_space
from app.embeddings import EmbeddingProvider
from app.models import Chunk, Document
from app.text import tokenize


def ingest_document(
    session: Session,
    provider: EmbeddingProvider,
    *,
    title: str,
    content: str,
    metadata: dict,
) -> Document:
    """Chunk, embed, and persist a document with its chunks in one unit.

    The caller owns the transaction: everything is flushed but not
    committed here, so an embedding or database failure leaves nothing
    half-ingested.

    Chunks without indexable tokens (punctuation-only windows) are not
    stored: they would embed to a zero vector, which cosine distance
    cannot rank, and full-text search cannot match them either. Raises
    ValueError when no chunk with indexable tokens remains.
    """
    settings = get_settings()

    chunks = [
        chunk
        for chunk in chunk_text(
            content,
            max_words=settings.chunk_max_words,
            overlap_words=settings.chunk_overlap_words,
        )
        if tokenize(chunk)
    ]
    if not chunks:
        raise ValueError("content contains no indexable tokens")

    embeddings = provider.embed(chunks)

    # Claimed under the row lock, in this same transaction: the chunks
    # below can only ever commit into an index whose stored embedding
    # space equals this provider's, so two providers can never mix — not
    # even across replicas. Raises EmbeddingSpaceMismatch otherwise.
    claim_embedding_space(session, provider.space)

    document = Document(title=title, content=content, doc_metadata=metadata)
    document.chunks = [
        Chunk(chunk_index=index, content=chunk, embedding=embedding)
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]

    session.add(document)
    session.flush()

    # Same transaction as the document write: the cache epoch bump commits
    # iff the ingest commits, so invalidation can never be lost — not even
    # while Redis is down (the epoch lives in PostgreSQL).
    bump_epoch(session)

    return document
