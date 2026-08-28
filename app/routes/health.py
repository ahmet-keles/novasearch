from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text

from app.cache import SearchCache, get_search_cache
from app.db import get_engine

router = APIRouter(tags=["health"])


@router.get("/health")
def health(
    response: Response,
    cache: SearchCache = Depends(get_search_cache),
) -> dict:
    database_up = _database_up()
    redis_up = cache.ping()

    if not (database_up and redis_up):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if database_up and redis_up else "degraded",
        "database": "up" if database_up else "down",
        "redis": "up" if redis_up else "down",
    }


def _database_up() -> bool:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
