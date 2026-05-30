# tests/test_core/test_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.models.resnet import ResNetClassifier, ResNetConfig
from app.core.training import cross_validate, train, train_one_epoch, validate_one_epoch
from app.preprocessing.reproducibility import set_seed


class TestTraining:
    """Tests for training functions."""

    def test_train_one_epoch_returns_finite_loss(self, tiny_dataloader, device):
        """Test that train_one_epoch returns finite loss."""
        set_seed(42)

        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        avg_loss = train_one_epoch(model, tiny_dataloader, criterion, optimizer, device)

        assert torch.isfinite(torch.tensor(avg_loss))
        assert avg_loss > 0

    def test_validate_one_epoch_returns_valid_metrics(self, tiny_dataloader, device):
        """Test validation returns valid metrics."""
        set_seed(42)

        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)
        criterion = nn.CrossEntropyLoss()

        avg_loss, accuracy, f1 = validate_one_epoch(model, tiny_dataloader, criterion, device, num_classes=2)

        assert torch.isfinite(torch.tensor(avg_loss))
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_overfit_single_batch(self, tiny_dataset, device):
        """
        Core sanity check: model can overfit a tiny dataset.
        If this fails, something is fundamentally broken.
        """
        set_seed(42)

        loader = DataLoader(tiny_dataset, batch_size=10, shuffle=False)

        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Get initial loss and accuracy
        model.train()
        initial_loss = None
        initial_acc = None
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            initial_loss = criterion(output, y).item()
            preds = output.argmax(dim=1)
            initial_acc = (preds == y).float().mean().item()
            break

        # Train for 50 epochs on same batch
        for epoch in range(50):
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        # Check final loss and accuracy
        model.eval()
        with torch.no_grad():
            output = model(x)
            final_loss = criterion(output, y).item()
            preds = output.argmax(dim=1)
            final_acc = (preds == y).float().mean().item()

        # Loss should decrease significantly (allow tolerance)
        assert final_loss < initial_loss * 0.5, f"Model failed to overfit: {initial_loss:.4f} -> {final_loss:.4f}"

        # Accuracy should improve
        assert final_acc > initial_acc, f"Accuracy did not improve: {initial_acc:.2%} -> {final_acc:.2%}"

    def test_train_returns_history(self, tiny_dataset, device):
        """Test full training loop returns history."""
        set_seed(42)

        train_loader = DataLoader(tiny_dataset, batch_size=5, shuffle=True)
        val_loader = DataLoader(tiny_dataset, batch_size=5, shuffle=False)

        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        trained_model, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            learning_rate=0.01,
            device=device,
            num_classes=2,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_accuracy" in history
        assert "val_f1" in history
        assert len(history["train_loss"]) == 3

    def test_train_with_patience_stops_early(self, tiny_dataset, device):
        """Early stopping with patience=2 stops before num_epochs."""
        set_seed(42)

        train_loader = DataLoader(tiny_dataset, batch_size=5, shuffle=True)
        val_loader = DataLoader(tiny_dataset, batch_size=5, shuffle=False)

        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        _, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
            learning_rate=0.001,
            device=device,
            num_classes=2,
            patience=2,
        )

        # Should stop well before 100 epochs
        assert len(history["train_loss"]) < 100

    def test_train_with_patience_zero_runs_all_epochs(self, tiny_dataset, device):
        """patience=0 (default) runs all epochs without early stopping."""
        set_seed(42)

        train_loader = DataLoader(tiny_dataset, batch_size=5, shuffle=True)
        val_loader = DataLoader(tiny_dataset, batch_size=5, shuffle=False)

        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        _, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            learning_rate=0.001,
            device=device,
            num_classes=2,
            patience=0,
        )

        assert len(history["train_loss"]) == 5


class TestCrossValidate:
    """Tests for cross_validate()."""

    def test_cross_validate_returns_expected_keys(self, tiny_dataset, device):
        """cross_validate returns mean_f1, std_f1, mean_accuracy, std_accuracy."""
        set_seed(42)

        config = ResNetConfig(num_classes=2, pretrained=False)

        results = cross_validate(
            model_config=config,
            dataset=tiny_dataset,
            num_folds=2,
            num_epochs=2,
            learning_rate=0.01,
            batch_size=5,
            device=device,
            num_classes=2,
        )

        assert "mean_f1" in results
        assert "std_f1" in results
        assert "mean_accuracy" in results
        assert "std_accuracy" in results

    def test_cross_validate_metrics_in_range(self, tiny_dataset, device):
        """Mean metrics should be in [0, 1]."""
        set_seed(42)

        config = ResNetConfig(num_classes=2, pretrained=False)

        results = cross_validate(
            model_config=config,
            dataset=tiny_dataset,
            num_folds=2,
            num_epochs=2,
            learning_rate=0.01,
            batch_size=5,
            device=device,
            num_classes=2,
        )

        assert 0.0 <= results["mean_f1"] <= 1.0
        assert 0.0 <= results["mean_accuracy"] <= 1.0
