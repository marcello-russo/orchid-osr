"""Tests for SupConLoss."""

import torch

from app.core.losses import SupConLoss


def test_supcon_basic():
    loss_fn = SupConLoss(temperature=0.07)
    features = torch.randn(8, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(features, labels)
    assert loss > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_supcon_same_class():
    loss_fn = SupConLoss(temperature=0.07)
    features = torch.randn(6, 128)
    labels = torch.zeros(6, dtype=torch.long)
    loss = loss_fn(features, labels)
    assert loss > 0
    assert not torch.isnan(loss)


def test_supcon_all_different():
    loss_fn = SupConLoss(temperature=0.07)
    features = torch.randn(6, 128)
    labels = torch.arange(6, dtype=torch.long)
    loss = loss_fn(features, labels)
    assert loss == 0.0


def test_supcon_batch_1():
    loss_fn = SupConLoss(temperature=0.07)
    features = torch.randn(1, 128)
    labels = torch.zeros(1, dtype=torch.long)
    loss = loss_fn(features, labels)
    assert loss == 0.0
    assert not torch.isnan(loss)


def test_supcon_single_positive():
    loss_fn = SupConLoss(temperature=0.07)
    features = torch.randn(4, 128)
    labels = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(features, labels)
    assert loss == 0.0
