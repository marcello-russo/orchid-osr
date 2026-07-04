"""Tests for extraction.py — ExtractionBundle and extract_all."""

import numpy as np
import pytest
import torch


class MockModel(torch.nn.Module):
    """Mock model che si comporta come un modello addestrato."""

    def __init__(self, d_model: int = 8, n_classes: int = 6):
        super().__init__()
        self._backbone = torch.nn.Identity()
        self._head = torch.nn.Linear(d_model, n_classes)
        self._embedding_dim = d_model

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.randn(B, self._embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embeddings(x)
        return self._head(emb)


@pytest.fixture
def mock_model():
    return MockModel(d_model=8, n_classes=6)


def test_get_classifier_known_variants():
    """_get_classifier riconosce tutte e 3 le architetture."""
    from app.core.ncd.extraction import _get_classifier

    model = MockModel()
    assert callable(_get_classifier(model, "dinov2_small"))
    assert callable(_get_classifier(model, "dinov2_base"))

    # ResNet e ConvNeXt hanno _backbone
    model._backbone.fc = torch.nn.Linear(8, 6)
    assert callable(_get_classifier(model, "resnet18"))
    assert callable(_get_classifier(model, "resnet50"))

    model._backbone.classifier = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Flatten(), torch.nn.Linear(8, 6))
    assert callable(_get_classifier(model, "convnext_tiny"))
    assert callable(_get_classifier(model, "convnext_small"))


def test_get_classifier_unknown_raises():
    """_get_classifier su modello sconosciuto alza ValueError."""
    from app.core.ncd.extraction import _get_classifier

    model = MockModel()
    with pytest.raises(ValueError, match="Unknown model_variant"):
        _get_classifier(model, "unknown_arch")


def test_extraction_bundle_creation():
    """ExtractionBundle si crea correttamente."""
    from app.core.ncd.extraction import ExtractionBundle

    bundle = ExtractionBundle(
        train_embeddings=np.zeros((10, 8)),
        train_logits=None,
        train_labels=np.zeros(10, dtype=int),
        train_class_names=["A", "B"],
        test_embeddings=np.zeros((5, 8)),
        test_logits=None,
        test_labels=np.zeros(5, dtype=int),
        test_class_names=["A", "B"],
    )
    assert bundle.train_embeddings.shape == (10, 8)
    assert bundle.test_embeddings.shape == (5, 8)
    assert bundle.train_logits is None
    assert bundle.test_logits is None


def test_extraction_bundle_with_logits():
    """ExtractionBundle con logits."""
    from app.core.ncd.extraction import ExtractionBundle

    bundle = ExtractionBundle(
        train_embeddings=np.zeros((10, 8)),
        train_logits=np.zeros((10, 6)),
        train_labels=np.zeros(10, dtype=int),
        train_class_names=["A", "B"],
        test_embeddings=np.zeros((5, 8)),
        test_logits=np.zeros((5, 6)),
        test_labels=np.zeros(5, dtype=int),
        test_class_names=["A", "B"],
    )
    assert bundle.train_logits.shape == (10, 6)
    assert bundle.test_logits.shape == (5, 6)
