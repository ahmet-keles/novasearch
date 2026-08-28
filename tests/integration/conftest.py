"""Fixtures for tests that need real PostgreSQL and Redis.

Point NOVA_DATABASE_URL / NOVA_REDIS_URL at running services (locally:
`docker compose up -d --wait`, then the defaults in .env.example match).
Migrations run once per session; tables are truncated before every test.
"""

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import get_engine
from app.main import app


@pytest.fixture(scope="session")
def migrated_database() -> None:
    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture
def client(migrated_database: None) -> TestClient:
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE documents CASCADE"))

    return TestClient(app)
