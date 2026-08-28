from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(
        title="NovaSearch",
        description="Hybrid search engine: PostgreSQL full-text + pgvector semantic search.",
        version="0.1.0",
    )

    from app.routes import documents

    app.include_router(documents.router)

    return app


app = create_app()
