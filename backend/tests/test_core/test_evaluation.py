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

        expected_keys = [
            "accuracy", "f1_macro", "f1_micro",
            "precision_macro", "recall_macro",
            "f1_per_class", "confusion_matrix", "per_class_metrics",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

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

    def test_confusion_matrix_sums_to_samples(self, tiny_dataloader, device, class_names_4):
        """Confusion matrix entries should sum to total number of samples."""
        set_seed(42)

        config = ResNetConfig(num_classes=4, pretrained=False)
        model = ResNetClassifier(config).to(device)

        results = evaluate(model, tiny_dataloader, class_names_4, device)

        cm = results["confusion_matrix"]
        total = sum(sum(row) for row in cm)
        assert total == len(tiny_dataloader.dataset)

    def test_biased_model_confusion_matrix(self, tiny_dataset, device, class_names_2):
        """Model always predicting class 0 → all predictions in column 0."""
        set_seed(42)

        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        # Mock model to always predict class 0
        def mock_forward(x):
            batch_size = x.shape[0]
            logits = torch.zeros(batch_size, 2, device=x.device)
            logits[:, 0] = 10.0  # class 0 always wins
            return logits

        model.forward = mock_forward

        from torch.utils.data import DataLoader
        loader = DataLoader(tiny_dataset, batch_size=10, shuffle=False)
        results = evaluate(model, loader, class_names_2, device)

        cm = results["confusion_matrix"]
        # Column 0 should have all predictions (sum of each row)
        col_0_sum = sum(row[0] for row in cm)
        assert col_0_sum == len(tiny_dataset)
        # Column 1 should be all zeros
        col_1_sum = sum(row[1] for row in cm)
        assert col_1_sum == 0
