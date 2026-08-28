import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_session
from app.embeddings import EmbeddingProvider, get_embedding_provider
from app.ingestion import ingest_document
from app.models import Document
from app.schemas import DocumentCreate, DocumentCreated, DocumentRead

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=DocumentCreated)
def create_document(
    payload: DocumentCreate,
    session: Session = Depends(get_session),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentCreated:
    try:
        document = ingest_document(
            session,
            provider,
            title=payload.title,
            content=payload.content,
            metadata=payload.metadata,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error

    session.commit()

    return DocumentCreated(id=document.id, chunk_count=len(document.chunks))


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    document_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> DocumentRead:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")

    return DocumentRead(
        id=document.id,
        title=document.title,
        content=document.content,
        metadata=document.doc_metadata,
        chunk_count=len(document.chunks),
        created_at=document.created_at,
    )
