import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_health_reports_all_dependencies_up(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "up", "redis": "up"}
