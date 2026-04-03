# tests/test_core/test_training.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from app.core.models.resnet import ResNetConfig, ResNetClassifier
from app.core.training import train_one_epoch, validate_one_epoch, train
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

        avg_loss = train_one_epoch(
            model, tiny_dataloader, criterion, optimizer, device
        )

        assert torch.isfinite(torch.tensor(avg_loss))
        assert avg_loss > 0

    def test_validate_one_epoch_returns_valid_metrics(self, tiny_dataloader, device):
        """Test validation returns valid metrics."""
        set_seed(42)

        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)
        criterion = nn.CrossEntropyLoss()

        avg_loss, accuracy, f1 = validate_one_epoch(
            model, tiny_dataloader, criterion, device, num_classes=2
        )

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
        assert final_loss < initial_loss * 0.5, \
            f"Model failed to overfit: {initial_loss:.4f} -> {final_loss:.4f}"

        # Accuracy should improve
        assert final_acc > initial_acc, \
            f"Accuracy did not improve: {initial_acc:.2%} -> {final_acc:.2%}"

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
