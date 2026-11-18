# app/core/kfold/supcon_epochs.py
"""Supervised Contrastive Learning utilities: projection head, training loop, embedding extraction."""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from app.core.kfold.epochs import cleanup_gpu
from app.core.losses import SupConLoss
from app.core.models.base import ClassifierModel

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Single-layer projection head for SupCon. Discarded after Phase 1.

    Multi-grain features (2304-dim vs 768-dim) make a two-layer MLP
    too large for 4GB VRAM: Linear(2304, 2304) = 5.3M extra params.
    Single linear projection saves memory with minimal accuracy loss. [8]
    """

    def __init__(self, embedding_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_optimizer(
    model: ClassifierModel,
    proj_head: ProjectionHead,
    learning_rate: float,
    optimizer_type: str = "adam",
    weight_decay: float = 0.0,
):
    """Create optimizer that covers backbone + projection head."""
    params = list(model.parameters()) + list(proj_head.parameters())
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


def train_supcon_one_epoch(
    model: ClassifierModel,
    proj_head: ProjectionHead,
    train_loader,
    criterion: SupConLoss,
    optimizer,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
) -> float:
    model.train()
    proj_head.train()
    running_loss = 0.0
    num_samples = 0
    optimizer.zero_grad(set_to_none=True)

    prototypes = getattr(criterion, "prototypes", None)
    repulse_weight = getattr(criterion, "_repulse_weight", 0.0)

    for batch_idx, (inputs, labels, _) in enumerate(train_loader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        embeddings = model.get_embeddings(inputs)
        projected = proj_head(embeddings)
        proj_norm = torch.norm(projected, dim=1, keepdim=True)
        projected = projected / (proj_norm + 1e-8)

        loss = criterion(projected, labels) / gradient_accumulation_steps

        if prototypes is not None and repulse_weight > 0:
            repulse = prototypes.repulse_loss() * repulse_weight / gradient_accumulation_steps
            loss = loss + repulse

        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(proj_head.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * gradient_accumulation_steps * inputs.size(0)
        num_samples += inputs.size(0)

    return running_loss / max(num_samples, 1)


def validate_supcon(
    model: ClassifierModel,
    val_loader,
    device: torch.device,
) -> tuple[float, float]:
    """Validate SupCon phase using KNN (k=1) accuracy on embeddings."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            embeddings = model.get_embeddings(inputs)
            embeddings = F.normalize(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    # KNN k=1: for each sample, find nearest neighbor (excluding self)
    sim = torch.matmul(all_embeddings, all_embeddings.T)
    sim.fill_diagonal_(-float("inf"))
    _, nn_idx = sim.max(dim=1)
    nn_preds = all_labels[nn_idx]
    knn_acc = (nn_preds == all_labels).float().mean().item()

    avg_norm = all_embeddings.norm(dim=1).mean().item()
    return knn_acc, avg_norm


def train_supcon(
    model: ClassifierModel,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_dir: str | None = None,
    proj_dim: int = 128,
    temperature: float = 0.07,
    patience: int = 0,
    gradient_accumulation_steps: int = 1,
    resume: bool = False,
    optimizer_type: str = "adam",
    weight_decay: float = 0.0,
    use_prototypes: bool = False,
    repulse_weight: float = 0.05,
    pair_weights: dict | None = None,
) -> tuple[ClassifierModel, dict]:
    """Phase 1: train backbone + projection head with SupCon loss.

    If use_prototypes=True, adds orthogonal prototype vectors (SpHOR-style)
    with a repulsion loss that encourages class prototypes to be orthogonal
    on the unit sphere, improving open-set separability.

    If pair_weights is provided, applies hard negative mining: specific class
    pairs receive higher loss weight to force separation of difficult pairs.

    Returns (model with best backbone weights, metrics dict).
    """

    model = model.to(device)
    proj_head = ProjectionHead(model.embedding_dim, proj_dim).to(device)

    criterion = SupConLoss(temperature=temperature, pair_weights=pair_weights)

    if use_prototypes:
        from app.core.losses import SphericalPrototypes

        prototypes = SphericalPrototypes(model.num_classes, proj_dim).to(device)
        criterion.prototypes = prototypes
        criterion._repulse_weight = repulse_weight
        logger.info(f"  Spherical prototypes: {model.num_classes} classes, repulse={repulse_weight}")

    optimizer = _make_optimizer(model, proj_head, learning_rate, optimizer_type, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    best_knn = 0.0
    best_backbone_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    start_epoch = 0

    history = {"train_loss": [], "val_knn_acc": [], "val_embedding_norm": []}

    # Resume
    ckpt_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "last_ckpt_supcon.pt")

    if resume and ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        proj_head.load_state_dict(ckpt["proj_head_state_dict"])
        if use_prototypes and "prototypes_state_dict" in ckpt:
            prototypes.load_state_dict(ckpt["prototypes_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_knn = ckpt["best_knn"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        history = ckpt["history"]
        best_backbone_wts = copy.deepcopy(model.state_dict())
        logger.info(f"SupCon resumed from epoch {start_epoch}, best KNN: {best_knn:.4f}")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"[SupCon] Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_supcon_one_epoch(
            model, proj_head, train_loader, criterion, optimizer, device, gradient_accumulation_steps
        )

        knn_acc, avg_norm = validate_supcon(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_knn_acc"].append(knn_acc)
        history["val_embedding_norm"].append(avg_norm)

        logger.info(f"[SupCon] Train Loss: {train_loss:.4f} | Val KNN Acc: {knn_acc:.4f} | Emb Norm: {avg_norm:.4f}")

        if knn_acc > best_knn:
            best_knn = knn_acc
            best_backbone_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            logger.info(f"[SupCon] New best KNN: {knn_acc:.4f}")
        else:
            epochs_no_improve += 1

        if ckpt_path:
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "proj_head_state_dict": proj_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_knn": best_knn,
                "epochs_no_improve": epochs_no_improve,
                "history": history,
            }
            if use_prototypes:
                ckpt_data["prototypes_state_dict"] = prototypes.state_dict()
            tmp_path = ckpt_path + ".tmp"
            torch.save(ckpt_data, tmp_path)
            os.replace(tmp_path, ckpt_path)

        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"[SupCon] Early stopping at epoch {epoch + 1}")
            break

        scheduler.step(knn_acc)

    model.load_state_dict(best_backbone_wts)
    logger.info(f"[SupCon] Phase 1 complete. Best KNN Acc: {best_knn:.4f}")

    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    del proj_head
    cleanup_gpu()

    return model, {
        "best_knn_acc": best_knn,
        "supcon_history": history,
    }


def extract_embeddings(
    model: ClassifierModel,
    data_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract embeddings for an entire dataset.

    Returns
    -------
        (embeddings, labels, paths) — embeddings is (N, embedding_dim) numpy array.

    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for inputs, labels, paths in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            embeddings = model.get_embeddings(inputs)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            all_paths.extend(paths)

    return (
        np.concatenate(all_embeddings, axis=0),
        np.array(all_labels),
        all_paths,
    )
