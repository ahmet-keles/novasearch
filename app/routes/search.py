from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_session
from app.embeddings import EmbeddingProvider, get_embedding_provider
from app.schemas import SearchResponse, SearchResult
from app.search import SearchMode, hybrid_search, keyword_search, semantic_search

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, max_length=1000),
    mode: SearchMode = SearchMode.hybrid,
    limit: int = Query(default=10, ge=1, le=50),
    session: Session = Depends(get_session),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> SearchResponse:
    if mode is SearchMode.semantic:
        hits = semantic_search(session, provider, q, limit)
    elif mode is SearchMode.keyword:
        hits = keyword_search(session, q, limit)
    else:
        hits = hybrid_search(session, provider, q, limit)

    return SearchResponse(
        query=q,
        mode=mode.value,
        results=[SearchResult(**vars(hit)) for hit in hits],
    )
