from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.cache import SearchCache, get_search_cache
from app.db import get_session
from app.embeddings import EmbeddingProvider, get_embedding_provider
from app.schemas import SearchResponse, SearchResult
from app.search import SearchMode, hybrid_search, keyword_search, semantic_search
from app.text import tokenize

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, max_length=1000),
    mode: SearchMode = SearchMode.hybrid,
    limit: int = Query(default=10, ge=1, le=50),
    session: Session = Depends(get_session),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
    cache: SearchCache = Depends(get_search_cache),
) -> SearchResponse:
    # Punctuation/whitespace-only queries embed to a zero vector (cosine
    # distance undefined) and match no tsquery: nothing can search them,
    # so say so instead of pretending, in every mode.
    if not tokenize(q):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="query contains no indexable tokens",
        )

    # The version is captured once and reused for the write below, so a
    # search racing an ingestion can only write to the already-retired
    # namespace — never cache stale results under the new one.
    version = cache.current_version()

    cached = cache.get(version=version, mode=mode.value, query=q, limit=limit)
    if cached is not None:
        return SearchResponse(**cached)

    if mode is SearchMode.semantic:
        hits = semantic_search(session, provider, q, limit)
    elif mode is SearchMode.keyword:
        hits = keyword_search(session, q, limit)
    else:
        hits = hybrid_search(session, provider, q, limit)

    response = SearchResponse(
        query=q,
        mode=mode.value,
        results=[SearchResult(**vars(hit)) for hit in hits],
    )

    cache.put(
        version=version,
        mode=mode.value,
        query=q,
        limit=limit,
        payload=response.model_dump(mode="json"),
    )

    return response
