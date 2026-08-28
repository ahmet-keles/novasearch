from sqlalchemy.orm import Session

from app.chunking import chunk_text
from app.config import get_settings
from app.embeddings import EmbeddingProvider
from app.models import Chunk, Document


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

    Raises ValueError when the content yields no chunks (e.g. only
    punctuation or whitespace).
    """
    settings = get_settings()

    chunks = chunk_text(
        content,
        max_words=settings.chunk_max_words,
        overlap_words=settings.chunk_overlap_words,
    )
    if not chunks:
        raise ValueError("content produced no indexable chunks")

    embeddings = provider.embed(chunks)

    document = Document(title=title, content=content, doc_metadata=metadata)
    document.chunks = [
        Chunk(chunk_index=index, content=chunk, embedding=embedding)
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]

    session.add(document)
    session.flush()

    return document
