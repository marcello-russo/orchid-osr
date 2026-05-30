# tests/conftest.py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def deterministic_mode():
    """Ensure deterministic behavior for all tests."""
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_dataset():
    """Create a tiny dataset for fast testing (10 samples, 2 classes)."""
    return TinyDataset(n_samples=10, n_classes=2, img_size=224)


@pytest.fixture
def tiny_dataloader(tiny_dataset):
    """DataLoader for tiny dataset."""
    return DataLoader(tiny_dataset, batch_size=10, shuffle=False)


class TinyDataset(Dataset):
    """Minimal dataset for testing - random images with labels."""

    def __init__(self, n_samples: int = 10, n_classes: int = 2, img_size: int = 224):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.img_size = img_size

        # Generate random images (3 channels, img_size x img_size)
        self.images = torch.randn(n_samples, 3, img_size, img_size)
        # Generate labels
        self.labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], f"img_{idx}"


@pytest.fixture
def class_names_6():
    """Standard 6 orchid class names."""
    return ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]


@pytest.fixture
def class_names_4():
    """Internet dataset class names (4 classes)."""
    return ["O. exaltata", "O. garganica", "O. incubacea", "O. sphegodes"]


@pytest.fixture
def class_names_2():
    """Minimal 2-class names for fast tests."""
    return [
        "O. exaltata",
        "O. garganica",
    ]
