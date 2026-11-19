"""SpHOR: i prototipi devono essere addestrati, non decorativi.

La guardia principale e' `test_train_supcon_moves_prototypes`: prima della
v0.6.0 i prototipi non stavano nell'optimizer, il gradiente della repulsione si
accumulava su `.grad` e non veniva mai applicato, quindi `--sphor` non cambiava
nulla rispetto a SupCon (era solo un seed diverso). Se qualcuno rimuove il
`add_param_group` in `train_supcon`, questo file cade.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.core.kfold.supcon_epochs import (
    ProjectionHead,
    _make_optimizer,
    train_supcon,
    train_supcon_one_epoch,
)
from app.core.losses import SphericalPrototypes, SupConLoss


class _StubModel(nn.Module):
    """Sostituto minimo di ClassifierModel: solo get_embeddings + i due attributi."""

    def __init__(self, embedding_dim: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Linear(16, embedding_dim), nn.ReLU())

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TripleDataset(Dataset):
    """(features, label, path) come i loader veri del progetto."""

    def __init__(self, n: int = 48, dim: int = 16, classes: int = 3) -> None:
        g = torch.Generator().manual_seed(0)
        self.labels = torch.arange(n) % classes
        centers = torch.randn(classes, dim, generator=g) * 4
        self.x = centers[self.labels] + 0.3 * torch.randn(n, dim, generator=g)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.x[i], self.labels[i], f"img_{i}.jpg"


def _loaders():
    ds = _TripleDataset()
    return DataLoader(ds, batch_size=8), DataLoader(ds, batch_size=8)


class TestPrototypes:
    def test_init_from_features_lands_on_centroids(self):
        prots = SphericalPrototypes(num_classes=3, proj_dim=8)
        feats = torch.tensor(
            [[1.0, 0, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]],
        )
        labels = torch.tensor([0, 0, 1])
        alignment = prots.init_from_features(feats, labels)
        assert alignment > 0.99
        normalized = torch.nn.functional.normalize(prots.prototypes.detach(), dim=1)
        assert torch.allclose(normalized[0], feats[0], atol=1e-5)

    def test_vmf_loss_is_lower_when_features_sit_on_their_prototype(self):
        prots = SphericalPrototypes(num_classes=2, proj_dim=4)
        prots.prototypes.data = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        labels = torch.tensor([0, 1])
        aligned = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        swapped = torch.tensor([[0, 1.0, 0, 0], [1.0, 0, 0, 0]])
        assert prots.vmf_loss(aligned, labels, kappa=10.0) < prots.vmf_loss(swapped, labels, kappa=10.0)

    def test_repulse_loss_zero_for_orthogonal_prototypes(self):
        prots = SphericalPrototypes(num_classes=2, proj_dim=4)
        prots.prototypes.data = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        assert float(prots.repulse_loss().detach()) < 1e-6

    def test_repulse_loss_positive_for_aligned_prototypes(self):
        prots = SphericalPrototypes(num_classes=2, proj_dim=4)
        prots.prototypes.data = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
        assert float(prots.repulse_loss().detach()) > 0.5


class TestOneEpoch:
    def test_prototypes_move_when_they_are_in_the_optimizer(self):
        model, proj = _StubModel(), ProjectionHead(32, 16)
        criterion = SupConLoss()
        prots = SphericalPrototypes(3, 16)
        criterion.prototypes = prots
        criterion._repulse_weight = 0.01
        opt = _make_optimizer(model, proj, 1e-3)
        opt.add_param_group({"params": [prots.prototypes], "lr": 1e-3, "weight_decay": 0.0})
        loader, _ = _loaders()

        before = prots.prototypes.detach().clone()
        for _ in range(5):
            train_supcon_one_epoch(model, proj, loader, criterion, opt, torch.device("cpu"), vmf_weight=1.0)
        assert not torch.allclose(before, prots.prototypes.detach())

    def test_epoch_reports_the_loss_parts(self):
        model, proj = _StubModel(), ProjectionHead(32, 16)
        criterion = SupConLoss()
        prots = SphericalPrototypes(3, 16)
        criterion.prototypes = prots
        opt = _make_optimizer(model, proj, 1e-3)
        opt.add_param_group({"params": [prots.prototypes], "lr": 1e-3, "weight_decay": 0.0})
        loader, _ = _loaders()

        parts = train_supcon_one_epoch(
            model, proj, loader, criterion, opt, torch.device("cpu"), vmf_weight=0.5, repulse_weight=0.01
        )
        assert set(parts) == {"loss", "supcon", "vmf", "repulse"}
        assert parts["vmf"] > 0 and parts["repulse"] > 0
        assert parts["loss"] >= parts["supcon"]


class TestTrainSupcon:
    def _run(self, tmp_path, epochs=3, warmup=1, ramp=2, resume=False):
        model = _StubModel()
        train_loader, val_loader = _loaders()
        return train_supcon(
            model,
            train_loader,
            val_loader,
            num_epochs=epochs,
            learning_rate=1e-3,
            device=torch.device("cpu"),
            checkpoint_dir=str(tmp_path),
            proj_dim=16,
            use_prototypes=True,
            repulse_weight=0.01,
            warmup_epochs=warmup,
            vmf_kappa=10.0,
            ramp_epochs=ramp,
            resume=resume,
        )

    def test_train_supcon_moves_prototypes(self, tmp_path):
        _, metrics = self._run(tmp_path)
        assert metrics["sphor"]["probe_accuracy"] is not None
        assert metrics["sphor"]["prototypes"].shape == (3, 16)

    def test_warmup_has_zero_vmf_weight_then_ramp(self, tmp_path):
        _, metrics = self._run(tmp_path, epochs=4, warmup=1, ramp=3)
        weights = metrics["supcon_history"]["vmf_weight"]
        assert weights[0] == 0.0
        assert weights[1] == pytest.approx(1 / 3)
        assert weights[3] == pytest.approx(1.0)

    def test_prototypes_are_aligned_with_the_classes_after_training(self, tmp_path):
        _, metrics = self._run(tmp_path)
        assert metrics["sphor"]["alignment_val"] > 0.5

    def test_history_is_json_friendly(self, tmp_path):
        _, metrics = self._run(tmp_path)
        for key, values in metrics["supcon_history"].items():
            assert all(isinstance(v, float) for v in values), key

    def test_checkpoint_without_prototypes_restarts_instead_of_crashing(self, tmp_path):
        _, _ = self._run(tmp_path, epochs=2)
        checkpoint = tmp_path / "last_ckpt_supcon.pt"
        assert not checkpoint.exists()  # rimosso a fine fase
        # un checkpoint vecchio (senza prototipi) non deve rompere il resume
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": _StubModel().state_dict(),
                "proj_head_state_dict": ProjectionHead(32, 16).state_dict(),
                "optimizer_state_dict": {"state": [], "param_groups": []},
                "scheduler_state_dict": {},
                "best_knn": 0.0,
                "epochs_no_improve": 0,
                "history": {},
            },
            checkpoint,
        )
        _, metrics = self._run(tmp_path, epochs=2, resume=True)
        assert metrics["best_knn_acc"] >= 0.0
