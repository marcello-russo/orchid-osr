"""Cosa vede chi non ha un account: anteprima del dataset e limiti sulla demo.

Il dataset è privato su HuggingFace: il proxy è l'unica via alle immagini, quindi
il limite dell'anteprima deve stare nel backend, non solo nella UI.
"""

from unittest.mock import patch
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

import app.api.routes.dataset as dataset_routes
from app.api.main import app
from app.api.routes.auth import limiter
from app.core.auth import create_jwt

client = TestClient(app)
AUTH_HEADER = {"Authorization": f"Bearer {create_jwt('test', 'user')}"}
CLASS = quote("O. exaltata")

MOCK_LISTING = {
    "classes": {"O. exaltata": {"train": 5, "test": 1}},
    "files": {
        "train": {"O. exaltata": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]},
        "test": {"O. exaltata": ["t1.jpg"]},
    },
    "total_images": 6,
}


@pytest.fixture(autouse=True)
def _fresh_preview_cache():
    dataset_routes._preview_cache = None
    yield
    dataset_routes._preview_cache = None


class TestDatasetPreview:
    def test_classes_are_public(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            assert client.get("/api/dataset/classes").status_code == 200

    def test_anonymous_sees_three_images_per_class(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            response = client.get(f"/api/dataset/classes/{CLASS}/images?limit=50")
        assert response.status_code == 200
        body = response.json()
        assert len(body["images"]) == 3
        # il totale resta visibile: l'anteprima è sulle foto, non sui numeri
        assert body["total"] == 5

    def test_authenticated_sees_the_whole_class(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            response = client.get(f"/api/dataset/classes/{CLASS}/images?limit=50", headers=AUTH_HEADER)
        assert response.status_code == 200
        assert len(response.json()["images"]) == 5

    def test_anonymous_cannot_fetch_a_non_preview_image(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            response = client.get(f"/api/dataset/images/train/{CLASS}/d.jpg")
        assert response.status_code == 403
        assert "Anteprima" in response.json()["detail"]


class TestClassifyRateLimit:
    def test_the_demo_is_throttled_per_ip(self):
        """`/classify` è pubblica: senza limite chiunque può far girare il modello."""
        # il conftest disattiva il limiter per gli altri test: qui serve acceso
        limiter.reset()
        limiter.enabled = True
        payload = {"image_base64": "x", "model_type": "modello-inesistente"}
        try:
            codes = [client.post("/classify", json=payload).status_code for _ in range(20)]
            assert codes == [400] * 20  # validazione prima del modello, nessuna inferenza
            assert client.post("/classify", json=payload).status_code == 429
        finally:
            limiter.enabled = False
            limiter.reset()
