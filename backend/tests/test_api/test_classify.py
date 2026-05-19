# tests/test_api/test_classify.py
import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app

client = TestClient(app)


def create_test_image_base64() -> str:
    """Create a simple test image encoded as base64."""
    img = Image.new('RGB', (224, 224), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


class TestClassifyEndpoint:
    """Tests for classify endpoint."""

    def test_classify_returns_200(self):
        """Test classify endpoint returns 200 with valid image."""
        image_base64 = create_test_image_base64()

        response = client.post(
            "/classify",
            json={"image_base64": image_base64, "model_type": "resnet18"}
        )

        assert response.status_code == 200

    def test_classify_response_structure(self):
        """Test classify response has required fields."""
        image_base64 = create_test_image_base64()

        response = client.post(
            "/classify",
            json={"image_base64": image_base64}
        )
        data = response.json()

        assert "predicted_class" in data
        assert "predicted_class_id" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_classify_confidence_range(self):
        """Test confidence is between 0 and 1."""
        image_base64 = create_test_image_base64()

        response = client.post(
            "/classify",
            json={"image_base64": image_base64}
        )
        data = response.json()

        assert 0.0 <= data["confidence"] <= 1.0

    def test_classify_probabilities_sum_to_one(self):
        """Test probabilities sum to approximately 1."""
        image_base64 = create_test_image_base64()

        response = client.post(
            "/classify",
            json={"image_base64": image_base64}
        )
        data = response.json()

        probs = list(data["probabilities"].values())
        assert abs(sum(probs) - 1.0) < 0.01

    def test_classify_invalid_base64_returns_400(self):
        """Test invalid base64 returns 400."""
        response = client.post(
            "/classify",
            json={"image_base64": "not_valid_base64!!!"}
        )

        assert response.status_code == 400
