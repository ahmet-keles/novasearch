"""Fixtures for tests that need real PostgreSQL and Redis.

Point NOVA_DATABASE_URL / NOVA_REDIS_URL at running services (locally:
`docker compose up -d --wait`, then the defaults in .env.example match).
Migrations run once per session; tables are truncated before every test.
"""

import pytest
import redis
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from alembic import command
from app.config import get_settings
from app.db import get_engine
from app.main import app


@pytest.fixture(scope="session")
def migrated_database() -> None:
    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture(scope="session")
def redis_client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


@pytest.fixture
def client(migrated_database: None, redis_client: redis.Redis) -> TestClient:
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE documents CASCADE"))
        # Cleared index -> unclaimed embedding space, same invariant the
        # migrations keep: each test adopts its provider's space afresh.
        connection.execute(
            text(
                "UPDATE embedding_space"
                " SET provider = NULL, model_name = NULL, dimension = NULL"
            )
        )
    redis_client.flushdb()

    return TestClient(app)
