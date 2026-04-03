# tests/test_core/test_evaluation.py
import pytest
import torch
from app.core.models.resnet import ResNetConfig, ResNetClassifier
from app.core.evaluation import evaluate
from app.preprocessing.reproducibility import set_seed


class TestEvaluation:
    """Tests for evaluation functions."""

    def test_evaluate_returns_correct_keys(self, tiny_dataloader, device, class_names_4):
        """Test that evaluate returns all expected keys."""
        set_seed(42)

        config = ResNetConfig(num_classes=4, pretrained=False)
        model = ResNetClassifier(config).to(device)

        results = evaluate(model, tiny_dataloader, class_names_4, device)

        assert "accuracy" in results
        assert "f1_macro" in results
        assert "precision_macro" in results
        assert "recall_macro" in results
        assert "f1_per_class" in results
        assert "confusion_matrix" in results
        assert "per_class_metrics" in results

    def test_evaluate_accuracy_range(self, tiny_dataloader, device, class_names_4):
        """Test that accuracy is in valid range."""
        set_seed(42)

        config = ResNetConfig(num_classes=4, pretrained=False)
        model = ResNetClassifier(config).to(device)

        results = evaluate(model, tiny_dataloader, class_names_4, device)

        assert 0.0 <= results["accuracy"] <= 1.0

    def test_evaluate_f1_range(self, tiny_dataloader, device, class_names_4):
        """Test that F1 scores are in valid range."""
        set_seed(42)

        config = ResNetConfig(num_classes=4, pretrained=False)
        model = ResNetClassifier(config).to(device)

        results = evaluate(model, tiny_dataloader, class_names_4, device)

        assert 0.0 <= results["f1_macro"] <= 1.0

        for class_name, f1 in results["f1_per_class"].items():
            assert 0.0 <= f1 <= 1.0, f"F1 for {class_name} out of range: {f1}"

    def test_evaluate_confusion_matrix_shape(self, tiny_dataloader, device, class_names_4):
        """Test that confusion matrix has correct shape."""
        set_seed(42)

        config = ResNetConfig(num_classes=4, pretrained=False)
        model = ResNetClassifier(config).to(device)

        results = evaluate(model, tiny_dataloader, class_names_4, device)

        cm = results["confusion_matrix"]
        n_classes = len(class_names_4)

        assert len(cm) == n_classes
        assert all(len(row) == n_classes for row in cm)

    def test_evaluate_per_class_metrics_structure(self, tiny_dataloader, device, class_names_4):
        """Test that per-class metrics have correct structure."""
        set_seed(42)

        config = ResNetConfig(num_classes=4, pretrained=False)
        model = ResNetClassifier(config).to(device)

        results = evaluate(model, tiny_dataloader, class_names_4, device)

        per_class = results["per_class_metrics"]

        for class_name in class_names_4:
            assert class_name in per_class
            assert "precision" in per_class[class_name]
            assert "recall" in per_class[class_name]
            assert "f1" in per_class[class_name]
            assert "support" in per_class[class_name]
