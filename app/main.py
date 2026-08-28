from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(
        title="NovaSearch",
        description="Hybrid search engine: PostgreSQL full-text + pgvector semantic search.",
        version="0.1.0",
    )

    from app.routes import documents, health, search

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)

    return app


app = create_app()
