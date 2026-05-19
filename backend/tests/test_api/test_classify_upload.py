# tests/test_api/test_classify_upload.py
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app

client = TestClient(app)


def create_test_image_bytes() -> BytesIO:
    """Create a PNG image as BytesIO for upload."""
    img = Image.new("RGB", (256, 512), color=(100, 150, 80))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestClassifyUploadEndpoint:
    """Tests for POST /classify/upload."""

    def test_upload_returns_200(self):
        buf = create_test_image_bytes()
        response = client.post(
            "/classify/upload",
            files={"file": ("test.png", buf, "image/png")},
        )
        assert response.status_code == 200

    def test_upload_response_structure(self):
        buf = create_test_image_bytes()
        response = client.post(
            "/classify/upload",
            files={"file": ("test.png", buf, "image/png")},
        )
        data = response.json()

        assert "top_predictions" in data
        assert "predicted_species" in data
        assert "confidence" in data
        assert isinstance(data["top_predictions"], list)
        assert len(data["top_predictions"]) > 0

    def test_upload_prediction_fields(self):
        buf = create_test_image_bytes()
        response = client.post(
            "/classify/upload",
            files={"file": ("test.png", buf, "image/png")},
        )
        pred = response.json()["top_predictions"][0]

        assert "species" in pred
        assert "confidence" in pred
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_upload_confidence_range(self):
        buf = create_test_image_bytes()
        response = client.post(
            "/classify/upload",
            files={"file": ("test.png", buf, "image/png")},
        )
        data = response.json()

        assert 0.0 <= data["confidence"] <= 1.0

    def test_upload_with_invalid_model_id_returns_error(self):
        buf = create_test_image_bytes()
        with pytest.raises(ValueError, match="No weights found"):
            client.post(
                "/classify/upload",
                files={"file": ("test.png", buf, "image/png")},
                data={"model_id": "nonexistent_model"},
            )

    def test_upload_no_file_returns_422(self):
        response = client.post("/classify/upload")
        assert response.status_code == 422


class TestExplainEndpoint:
    """Tests for POST /classify/explain (slow — requires model + captum)."""

    @pytest.mark.slow
    def test_explain_returns_200(self):
        buf = create_test_image_bytes()
        response = client.post(
            "/classify/explain",
            files={"file": ("test.png", buf, "image/png")},
        )
        assert response.status_code == 200

    @pytest.mark.slow
    def test_explain_response_structure(self):
        buf = create_test_image_bytes()
        response = client.post(
            "/classify/explain",
            files={"file": ("test.png", buf, "image/png")},
        )
        data = response.json()

        assert "heatmap_base64" in data
        assert "predicted_species" in data
        assert "confidence" in data
        assert isinstance(data["heatmap_base64"], str)
        assert len(data["heatmap_base64"]) > 100


class TestModelsEndpoint:
    """Tests for GET /classify/models."""

    def test_models_returns_200(self):
        response = client.get("/classify/models")
        assert response.status_code == 200

    def test_models_response_structure(self):
        response = client.get("/classify/models")
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)

    def test_models_items_have_fields(self):
        response = client.get("/classify/models")
        data = response.json()

        if data["models"]:
            m = data["models"][0]
            assert "id" in m
            assert "name" in m
            assert "model_variant" in m
