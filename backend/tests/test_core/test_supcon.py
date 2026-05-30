"""Tests for supcon training."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.core.models import create_model
from app.core.supcon import train_supcon


def _make_dummy_loader(batch_size=4, num_batches=3):
    X = torch.randn(num_batches * batch_size, 3, 224, 224)
    y = torch.randint(0, 3, (num_batches * batch_size,))
    ds = TensorDataset(X, y, torch.arange(len(X)))
    return DataLoader(ds, batch_size=batch_size)


@pytest.mark.slow
def test_train_supcon_basic():
    model = create_model("resnet18", num_classes=3, pretrained=False)
    train_loader = _make_dummy_loader(batch_size=4, num_batches=2)
    val_loader = _make_dummy_loader(batch_size=4, num_batches=1)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp:
        model_out, metrics = train_supcon(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            learning_rate=0.001,
            device=device,
            checkpoint_dir=tmp,
        )
    assert "knn_accuracy" in metrics
    assert len(metrics["knn_accuracy"]) == 2
    assert model_out is not None


@pytest.mark.slow
def test_train_supcon_no_nans():
    model = create_model("resnet18", num_classes=3, pretrained=False)
    train_loader = _make_dummy_loader(batch_size=2, num_batches=1)
    val_loader = _make_dummy_loader(batch_size=2, num_batches=1)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp:
        model_out, metrics = train_supcon(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            learning_rate=0.001,
            device=device,
            checkpoint_dir=tmp,
            patience=0,
        )
    assert not any(torch.isnan(torch.tensor(v)).any() for k, v in metrics.items() if isinstance(v, list) and v)


@pytest.mark.slow
def test_train_supcon_resume():
    model = create_model("resnet18", num_classes=3, pretrained=False)
    train_loader = _make_dummy_loader(batch_size=4, num_batches=2)
    val_loader = _make_dummy_loader(batch_size=4, num_batches=1)
    device = torch.device("cpu")

    tmp = Path(tempfile.mkdtemp())
    try:
        ckpt = tmp / "last_ckpt_supcon.pt"
        model, metrics1 = train_supcon(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            learning_rate=0.001,
            device=device,
            checkpoint_dir=str(tmp),
            resume=False,
        )
        assert ckpt.exists()

        model2 = create_model("resnet18", num_classes=3, pretrained=False)
        model2, metrics2 = train_supcon(
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=4,
            learning_rate=0.001,
            device=device,
            checkpoint_dir=str(tmp),
            resume=True,
        )
        assert len(metrics2.get("train_loss", [])) >= 2
    finally:
        import shutil

        shutil.rmtree(tmp)
