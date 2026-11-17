# tests/test_api/test_health.py
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_returns_200(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        """Test health response has required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_type" in data
        assert "timestamp" in data

    def test_health_status_is_healthy(self):
        """Test health status is 'healthy'."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_model_loaded_is_bool(self):
        """Test model_loaded is boolean."""
        response = client.get("/health")
        data = response.json()

        assert isinstance(data["model_loaded"], bool)


def test_static_mime_types_registered():
    """Python 3.11 non conosce .webp: il SPA server deve dichiararlo.

    Senza registrazione `FileResponse` serve le immagini come
    `application/octet-stream` (verificato in produzione su /figures/*.webp).
    """
    import mimetypes

    from app.api import main  # noqa: F401  (la registrazione avviene all'import)

    assert mimetypes.guess_type("figura.webp")[0] == "image/webp"
    assert mimetypes.guess_type("font.woff2")[0] == "font/woff2"
