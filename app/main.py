from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import get_sessionmaker, validate_embedding_dimension
from app.embeddings import get_embedding_provider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail fast on a provider/schema dimension mismatch instead of letting
    # the first ingest die with an opaque pgvector error.
    with get_sessionmaker()() as session:
        validate_embedding_dimension(session, get_embedding_provider())

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="NovaSearch",
        description="Hybrid search engine: PostgreSQL full-text + pgvector semantic search.",
        version="0.1.0",
        lifespan=lifespan,
    )

    from app.routes import documents, health, search

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)

    return app


app = create_app()
