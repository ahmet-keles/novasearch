from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import get_sessionmaker, validate_embedding_dimension
from app.embedding_space import validate_embedding_space
from app.embeddings import get_embedding_provider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail fast instead of letting the first query misbehave: the provider
    # must match the pgvector column dimension, and — dimensions being
    # equal is not enough — the embedding space of whatever the index
    # already holds. An unclaimed (empty) index accepts any provider.
    with get_sessionmaker()() as session:
        provider = get_embedding_provider()
        validate_embedding_dimension(session, provider)
        validate_embedding_space(session, provider.space)

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
