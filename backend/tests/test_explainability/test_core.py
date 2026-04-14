# tests/test_explainability/test_core.py
"""Tests for explainability core: inference() and get_percentages()."""
import importlib.util
import os
import pytest
import torch

from app.core.models.resnet import ResNetConfig, ResNetClassifier
from app.preprocessing.reproducibility import set_seed

# Load core module directly to avoid triggering matplotlib via __init__.py
_core_path = os.path.join(os.path.dirname(__file__), "..", "..", "app", "explainability", "core.py")
_core_path = os.path.abspath(_core_path)
_spec = importlib.util.spec_from_file_location("app.explainability.core", _core_path)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
inference = _core.inference
get_percentages = _core.get_percentages


class TestGetPercentages:
    """Tests for get_percentages (softmax * 100)."""

    def test_sums_to_100(self):
        """Percentages should sum to ~100."""
        values = torch.tensor([[1.0, 2.0, 3.0]])
        percentages = get_percentages(values)
        assert abs(percentages.sum().item() - 100.0) < 1e-4

    def test_all_equal_logits(self):
        """Equal logits should give ~equal percentages."""
        values = torch.tensor([[5.0, 5.0, 5.0]])
        percentages = get_percentages(values)
        assert abs(percentages[0, 0].item() - 100.0 / 3) < 1e-4
        assert abs(percentages[0, 1].item() - 100.0 / 3) < 1e-4
        assert abs(percentages[0, 2].item() - 100.0 / 3) < 1e-4

    def test_single_dominant_class(self):
        """Very large logit for one class → ~100% for that class."""
        values = torch.tensor([[100.0, 0.0, 0.0]])
        percentages = get_percentages(values)
        assert percentages[0, 0].item() > 99.9
        assert percentages[0, 1].item() < 0.1
        assert percentages[0, 2].item() < 0.1

    def test_output_shape(self):
        """Output shape matches input shape."""
        values = torch.randn(4, 6)
        percentages = get_percentages(values)
        assert percentages.shape == (4, 6)

    def test_all_positive(self):
        """All percentages should be positive."""
        values = torch.randn(2, 5)
        percentages = get_percentages(values)
        assert (percentages > 0).all()

    def test_batch_sums_to_100(self):
        """Each row in a batch should sum to ~100."""
        values = torch.randn(8, 4)
        percentages = get_percentages(values)
        row_sums = percentages.sum(dim=1)
        for s in row_sums:
            assert abs(s.item() - 100.0) < 1e-4


class TestInference:
    """Tests for inference()."""

    def test_returns_values_and_predicted(self, tiny_dataloader, device):
        """inference() returns (values, predicted) tuple."""
        set_seed(42)
        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        for x, _, _ in tiny_dataloader:
            single = x[0].to(device)  # pick one image [3, H, W]
            values, predicted = inference(model, single.unsqueeze(0), device)
            break

        assert isinstance(values, torch.Tensor)
        assert isinstance(predicted, torch.Tensor)
        assert values.shape == (1, 2)
        assert predicted.shape == (1,)

    def test_predicted_is_valid_class(self, tiny_dataloader, device):
        """Predicted class index must be in [0, num_classes)."""
        set_seed(42)
        config = ResNetConfig(num_classes=4, pretrained=False)
        model = ResNetClassifier(config).to(device)

        for x, _, _ in tiny_dataloader:
            single = x[0].to(device)
            values, predicted = inference(model, single.unsqueeze(0), device)
            break

        assert 0 <= predicted.item() < 4

    def test_model_in_eval_mode_after_inference(self, tiny_dataloader, device):
        """inference() sets model to eval mode."""
        set_seed(42)
        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)
        model.train()

        for x, _, _ in tiny_dataloader:
            single = x[0].to(device)
            inference(model, single.unsqueeze(0), device)
            break

        assert not model.training

    def test_consistent_predictions(self, tiny_dataloader, device):
        """Same input should give same prediction (deterministic in eval mode)."""
        set_seed(42)
        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        for x, _, _ in tiny_dataloader:
            single = x[0].to(device)
            _, pred1 = inference(model, single.unsqueeze(0), device)
            _, pred2 = inference(model, single.unsqueeze(0), device)
            assert pred1.item() == pred2.item()
            break
