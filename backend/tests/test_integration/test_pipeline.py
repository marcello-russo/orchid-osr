# tests/test_integration/test_pipeline.py
"""
End-to-end integration test using tiny dataset.

Tests the full pipeline: augmentation -> preprocessing -> training -> testing
using the modular core components.
"""
import pytest
import torch
import os
import shutil
from pathlib import Path

from app.core.models.resnet import ResNetConfig, ResNetClassifier
from app.core.training import train
from app.core.evaluation import evaluate
from app.preprocessing.dataset_tool import ImageFolderWithName, TensorDatasetWithName
from app.preprocessing.reproducibility import set_seed
from torch.utils.data import DataLoader
import torchvision.transforms as T


# Test dataset path (at project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATASET_DIR = PROJECT_ROOT / "test_dataset"


@pytest.fixture(scope="module")
def tiny_config():
    """Configuration for tiny dataset tests."""
    return ResNetConfig(
        variant="resnet18",
        num_classes=6,
        pretrained=False  # Faster for testing
    )


@pytest.fixture(scope="module")
def tiny_device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_transforms():
    """Transforms for tiny dataset."""
    return T.Compose([
        T.Resize((512, 256)),
        T.ToTensor(),
    ])


@pytest.fixture(scope="module")
def tiny_train_dataset(tiny_transforms):
    """Tiny training dataset."""
    train_dir = TEST_DATASET_DIR / "train"
    if not train_dir.exists():
        pytest.skip("test_dataset not found - extract from combined first")
    return ImageFolderWithName(str(train_dir), transform=tiny_transforms)


@pytest.fixture(scope="module")
def tiny_val_dataset(tiny_transforms):
    """Tiny validation dataset."""
    val_dir = TEST_DATASET_DIR / "val"
    if not val_dir.exists():
        pytest.skip("test_dataset not found")
    return ImageFolderWithName(str(val_dir), transform=tiny_transforms)


@pytest.fixture(scope="module")
def tiny_test_dataset(tiny_transforms):
    """Tiny test dataset."""
    test_dir = TEST_DATASET_DIR / "test"
    if not test_dir.exists():
        pytest.skip("test_dataset not found")
    return ImageFolderWithName(str(test_dir), transform=tiny_transforms)


class TestIntegrationPipeline:
    """Integration tests for the complete pipeline."""

    def test_tiny_dataset_exists(self):
        """Test that tiny dataset was created correctly."""
        assert TEST_DATASET_DIR.exists(), "Tiny dataset directory should exist"

        for split in ["train", "val", "test"]:
            split_dir = TEST_DATASET_DIR / split
            assert split_dir.exists(), f"{split} directory should exist"

            # Count images
            total = 0
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg")))
                    total += count
            assert total > 0, f"{split} should have images"

    def test_train_on_tiny_dataset(self, tiny_config, tiny_device, tiny_train_dataset, tiny_val_dataset):
        """Test training on tiny dataset."""
        set_seed(42)

        train_loader = DataLoader(tiny_train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(tiny_val_dataset, batch_size=4, shuffle=False)

        model = ResNetClassifier(tiny_config).to(tiny_device)

        # Train for just 2 epochs
        trained_model, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            learning_rate=0.01,
            device=tiny_device,
            num_classes=6,
        )

        # Verify training ran
        assert len(history["train_loss"]) == 2
        assert len(history["val_f1"]) == 2
        assert history["val_f1"][-1] > 0, "F1 should improve"

    def test_evaluate_on_tiny_dataset(self, tiny_config, tiny_device, tiny_test_dataset):
        """Test evaluation on tiny dataset."""
        set_seed(42)

        # Create a model with random weights
        model = ResNetClassifier(tiny_config).to(tiny_device)

        test_loader = DataLoader(tiny_test_dataset, batch_size=4, shuffle=False)

        class_names = ["O. exaltata", "O. garganica", "O. incubacea",
                       "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]

        results = evaluate(model, test_loader, class_names, tiny_device)

        # Verify evaluation results
        assert "accuracy" in results
        assert "f1_macro" in results
        assert "confusion_matrix" in results
        assert 0 <= results["accuracy"] <= 1

    def test_overfit_single_class(self, tiny_config, tiny_device, tiny_train_dataset):
        """Test that model can overfit on a single batch (sanity check)."""
        set_seed(42)

        # Get a single batch
        train_loader = DataLoader(tiny_train_dataset, batch_size=4, shuffle=False)
        val_loader = DataLoader(tiny_train_dataset, batch_size=4, shuffle=False)

        model = ResNetClassifier(tiny_config).to(tiny_device)

        # Train for many epochs on same data
        trained_model, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,
            learning_rate=0.1,  # Higher LR for faster overfitting
            device=tiny_device,
            num_classes=6,
        )

        # Loss should decrease significantly
        assert history["train_loss"][-1] < history["train_loss"][0], \
            "Training loss should decrease when overfitting"
