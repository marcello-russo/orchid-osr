import torch

from app.core.losses import SupConLoss


class TestSupConLoss:
    def test_forward_returns_scalar(self):
        loss_fn = SupConLoss(temperature=0.07)
        features = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(features, labels)
        assert loss.ndim == 0
        assert loss > 0

    def test_two_identical_samples_have_zero_loss(self):
        loss_fn = SupConLoss(temperature=0.07)
        f = torch.randn(1, 128).expand(2, -1)
        labels = torch.zeros(2, dtype=torch.long)
        loss = loss_fn(f, labels)
        assert loss < 0.01

    def test_different_classes_have_positive_loss(self):
        loss_fn = SupConLoss(temperature=0.07)
        features = torch.randn(4, 128)
        labels = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(features, labels)
        assert loss > 0

    def test_temperature_scales_loss(self):
        features = torch.randn(8, 128)
        labels = torch.tensor([0] * 4 + [1] * 4)
        loss_low = SupConLoss(temperature=0.07)(features, labels)
        loss_high = SupConLoss(temperature=1.0)(features, labels)
        assert loss_low > loss_high

    def test_large_batch(self):
        loss_fn = SupConLoss(temperature=0.07)
        features = torch.randn(64, 128)
        labels = torch.randint(0, 8, (64,))
        loss = loss_fn(features, labels)
        assert loss.ndim == 0
