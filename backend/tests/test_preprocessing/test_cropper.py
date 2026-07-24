from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from app.preprocessing.cropper import (
    DEFAULT_SCORE_THRESHOLD,
    crop_single_image,
    crop_to_bbox,
    detect_flower,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_image(size=(100, 200)):
    """Return a random RGB PIL image of *size* (width, height)."""
    return Image.fromarray((np.random.rand(*size[::-1], 3) * 255).astype(np.uint8))


def _make_mock_model(has_detection=True, score=0.9, bbox=(10, 20, 80, 90)):
    """Build a MagicMock that mimics a Faster R-CNN eval output."""
    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor([1.0])])

    if has_detection:
        boxes = torch.tensor([[*bbox]], dtype=torch.float32)
        scores = torch.tensor([score])
        labels = torch.tensor([1])
    else:
        boxes = torch.empty(0, 4)
        scores = torch.empty(0)
        labels = torch.empty(0, dtype=torch.int64)

    model.return_value = [{"boxes": boxes, "scores": scores, "labels": labels}]
    return model


# ---------------------------------------------------------------------------
# crop_to_bbox
# ---------------------------------------------------------------------------


class TestCropToBbox:
    """crop_to_bbox needs no model — pure PIL logic."""

    def test_basic_crop(self):
        img = _dummy_image((200, 100))
        bbox = (20, 10, 120, 60)
        cropped = crop_to_bbox(img, bbox)
        assert cropped.size == (100, 50)

    def test_output_pixels_match_region(self):
        img = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
        arr = np.array(img)
        arr[10:30, 20:60] = [255, 128, 0]
        img = Image.fromarray(arr)

        cropped = crop_to_bbox(img, (20, 10, 60, 30))
        c_arr = np.array(cropped)
        assert c_arr.shape == (20, 40, 3)
        assert tuple(c_arr[0, 0]) == (255, 128, 0)

    def test_with_padding(self):
        img = _dummy_image((200, 100))
        bbox = (50, 30, 150, 70)
        padded = crop_to_bbox(img, bbox, padding=10)
        unpadded = crop_to_bbox(img, bbox, padding=0)
        assert padded.size[0] > unpadded.size[0]
        assert padded.size[1] > unpadded.size[1]

    def test_padding_clamped_to_image_bounds(self):
        img = _dummy_image((100, 100))
        bbox = (0, 0, 20, 20)
        result = crop_to_bbox(img, bbox, padding=50)
        assert result.size == (70, 70)

    def test_padding_overflow_both_sides(self):
        img = _dummy_image((100, 100))
        bbox = (-10, -10, 110, 110)
        result = crop_to_bbox(img, bbox, padding=0)
        assert result.size == (100, 100)

    def test_zero_area_bbox(self):
        img = _dummy_image((100, 100))
        bbox = (50, 50, 50, 50)
        result = crop_to_bbox(img, bbox)
        assert result.size == (0, 0)

    def test_negative_coordinates_bbox(self):
        img = _dummy_image((200, 200))
        bbox = (-20, -10, 80, 90)
        result = crop_to_bbox(img, bbox, padding=0)
        assert result.size == (80, 90)
        assert np.array(result).shape == (90, 80, 3)


# ---------------------------------------------------------------------------
# detect_flower
# ---------------------------------------------------------------------------


class TestDetectFlower:
    """detect_flower depends on a model — we mock it."""

    def test_detection_found(self):
        bbox = (15, 25, 85, 95)
        model = _make_mock_model(has_detection=True, score=0.85, bbox=bbox)
        img = _dummy_image()
        result = detect_flower(model, img, score_threshold=0.5)
        assert result == bbox

    def test_no_detection(self):
        model = _make_mock_model(has_detection=False)
        img = _dummy_image()
        assert detect_flower(model, img) is None

    def test_below_threshold(self):
        model = _make_mock_model(has_detection=True, score=0.3)
        img = _dummy_image()
        assert detect_flower(model, img, score_threshold=0.5) is None

    def test_default_threshold(self):
        bbox = (0, 0, 50, 50)
        model = _make_mock_model(has_detection=True, score=0.5, bbox=bbox)
        img = _dummy_image()
        result = detect_flower(model, img, score_threshold=DEFAULT_SCORE_THRESHOLD)
        assert result == bbox

    def test_score_below_default(self):
        model = _make_mock_model(has_detection=True, score=0.49)
        img = _dummy_image()
        assert detect_flower(model, img, score_threshold=DEFAULT_SCORE_THRESHOLD) is None

    def test_model_called_with_correct_device(self):
        model = _make_mock_model()
        img = _dummy_image()
        detect_flower(model, img)
        call_args = model.call_args[0][0]
        assert isinstance(call_args, list)
        assert isinstance(call_args[0], torch.Tensor)


# ---------------------------------------------------------------------------
# crop_single_image
# ---------------------------------------------------------------------------


class TestCropSingleImage:
    """crop_single_image combines detection + cropping — model is mocked."""

    def test_detected(self, tmp_path):
        img_path = tmp_path / "flower.jpg"
        _dummy_image((200, 200)).save(img_path)

        model = _make_mock_model(has_detection=True, score=0.9, bbox=(20, 20, 100, 100))
        cropped, meta = crop_single_image(model, str(img_path))

        assert cropped is not None
        assert meta["detected"] is True
        assert not meta["fallback_used"]
        assert meta["bbox"] == (20, 20, 100, 100)
        assert cropped.size == (80, 80)

    def test_detected_with_padding(self, tmp_path):
        img_path = tmp_path / "flower.jpg"
        _dummy_image((200, 200)).save(img_path)

        model = _make_mock_model(has_detection=True, score=0.9, bbox=(20, 20, 100, 100))
        cropped, meta = crop_single_image(model, str(img_path), padding=15)

        assert meta["detected"] is True
        assert cropped.size == (110, 110)

    def test_fallback_full(self, tmp_path):
        img_path = tmp_path / "flower.jpg"
        orig = _dummy_image((200, 200))
        orig.save(img_path)

        model = _make_mock_model(has_detection=False)
        cropped, meta = crop_single_image(model, str(img_path), fallback="full")

        assert cropped is not None
        assert not meta["detected"]
        assert meta["fallback_used"]
        assert cropped.size == (200, 200)

    def test_fallback_skip(self, tmp_path):
        img_path = tmp_path / "flower.jpg"
        _dummy_image((100, 100)).save(img_path)

        model = _make_mock_model(has_detection=False)
        cropped, meta = crop_single_image(model, str(img_path), fallback="skip")

        assert cropped is None
        assert not meta["detected"]
        assert meta["fallback_used"]

    def test_metadata_path(self, tmp_path):
        img_path = tmp_path / "flower.jpg"
        _dummy_image((100, 100)).save(img_path)

        model = _make_mock_model(has_detection=False)
        _, meta = crop_single_image(model, str(img_path), fallback="skip")
        assert meta["path"] == str(img_path)

    def test_model_on_cpu_by_default(self, tmp_path):
        img_path = tmp_path / "flower.jpg"
        _dummy_image((100, 100)).save(img_path)

        model = _make_mock_model(has_detection=True, score=0.8, bbox=(10, 10, 50, 50))
        cropped, meta = crop_single_image(model, str(img_path))
        assert cropped is not None
        assert meta["detected"]
