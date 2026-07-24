import pytest

from app.preprocessing.reproducibility import set_seed


def _mock_create_model(model_variant, num_classes=6, pretrained=True, **kwargs):
    """Fast model without pretrained weights."""
    from app.core.models.resnet import ResNetClassifier, ResNetConfig

    config = ResNetConfig(num_classes=num_classes, pretrained=False, variant=model_variant)
    return ResNetClassifier(config)


class TestTrainFold:
    """Tests for train_fold()."""

    @pytest.fixture(autouse=True)
    def _no_pretrained(self, monkeypatch):
        """Avoid downloading pretrained ImageNet weights in tests."""
        monkeypatch.setattr("app.core.models.create_model", _mock_create_model)

    def test_train_fold_completes(self, tiny_dataloader, device, tmp_path):
        from app.core.kfold.training import train_fold

        ckpt = tmp_path / "best.pt"
        model, metrics = train_fold(
            model_variant="resnet18",
            num_classes=2,
            train_loader=tiny_dataloader,
            val_loader=tiny_dataloader,
            num_epochs=5,
            learning_rate=0.001,
            device=device,
            checkpoint_path=str(ckpt),
        )

        assert model is not None
        assert ckpt.exists()
        assert isinstance(metrics, dict)

    def test_train_fold_returns_best_metrics(self, tiny_dataloader, device, tmp_path):
        from app.core.kfold.training import train_fold

        ckpt = tmp_path / "best.pt"
        _, metrics = train_fold(
            model_variant="resnet18",
            num_classes=2,
            train_loader=tiny_dataloader,
            val_loader=tiny_dataloader,
            num_epochs=5,
            learning_rate=0.001,
            device=device,
            checkpoint_path=str(ckpt),
        )

        assert "best_f1" in metrics
        assert "best_accuracy" in metrics
        assert "best_epoch" in metrics
        assert "final_train_loss" in metrics
        assert "final_val_loss" in metrics
        assert "history" in metrics
        assert 0.0 <= metrics["best_f1"] <= 1.0
        assert 0.0 <= metrics["best_accuracy"] <= 1.0
        assert metrics["best_epoch"] >= 1

    def test_train_fold_with_few_epochs(self, tiny_dataloader, device, tmp_path):
        from app.core.kfold.training import train_fold

        ckpt = tmp_path / "best.pt"
        model, metrics = train_fold(
            model_variant="resnet18",
            num_classes=2,
            train_loader=tiny_dataloader,
            val_loader=tiny_dataloader,
            num_epochs=2,
            learning_rate=0.001,
            device=device,
            checkpoint_path=str(ckpt),
        )

        assert len(metrics["history"]["train_loss"]) == 2
        assert len(metrics["history"]["val_loss"]) == 2
        assert len(metrics["history"]["val_accuracy"]) == 2
        assert len(metrics["history"]["val_f1"]) == 2
        assert ckpt.exists()

    def test_train_fold_deterministic(self, tiny_dataloader, device, tmp_path):
        from app.core.kfold.training import train_fold

        set_seed(42)
        ckpt1 = tmp_path / "run1" / "best.pt"
        ckpt1.parent.mkdir(parents=True)
        _, m1 = train_fold(
            model_variant="resnet18",
            num_classes=2,
            train_loader=tiny_dataloader,
            val_loader=tiny_dataloader,
            num_epochs=3,
            learning_rate=0.001,
            device=device,
            checkpoint_path=str(ckpt1),
        )

        set_seed(42)
        ckpt2 = tmp_path / "run2" / "best.pt"
        ckpt2.parent.mkdir(parents=True)
        _, m2 = train_fold(
            model_variant="resnet18",
            num_classes=2,
            train_loader=tiny_dataloader,
            val_loader=tiny_dataloader,
            num_epochs=3,
            learning_rate=0.001,
            device=device,
            checkpoint_path=str(ckpt2),
        )

        assert m1["best_f1"] == m2["best_f1"]
        assert m1["best_accuracy"] == m2["best_accuracy"]
        assert m1["history"]["train_loss"] == m2["history"]["train_loss"]
        assert m1["history"]["val_f1"] == m2["history"]["val_f1"]
