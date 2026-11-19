# app/core/kfold/supcon_epochs.py
"""Supervised Contrastive Learning utilities: projection head, training loop, embedding extraction."""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
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
    vmf_weight: float = 0.0,
    vmf_kappa: float = 10.0,
    repulse_weight: float | None = None,
) -> dict[str, float]:
    """One Phase-1 epoch: SupCon, plus the SpHOR terms when they are active.

    ``vmf_weight`` is the ramp coefficient of the vMF hard-assignment term
    (0 during the warm-up, then 0 -> 1). ``repulse_weight`` overrides the
    weight stored on the criterion; ``None`` keeps it.

    Returns
    -------
        Mean per-sample values of ``loss``, ``supcon``, ``vmf`` and
        ``repulse`` over the epoch.

    """
    model.train()
    proj_head.train()
    running = {key: torch.zeros((), device=device) for key in ("loss", "supcon", "vmf", "repulse")}
    num_samples = 0
    optimizer.zero_grad(set_to_none=True)

    prototypes = getattr(criterion, "prototypes", None)
    if repulse_weight is None:
        repulse_weight = getattr(criterion, "_repulse_weight", 0.0)
    zero = torch.zeros((), device=device)

    for batch_idx, (inputs, labels, _) in enumerate(train_loader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        embeddings = model.get_embeddings(inputs)
        projected = proj_head(embeddings)
        proj_norm = torch.norm(projected, dim=1, keepdim=True)
        projected = projected / (proj_norm + 1e-8)

        supcon = criterion(projected, labels)
        loss = supcon / gradient_accumulation_steps

        vmf = zero
        if prototypes is not None and vmf_weight > 0:
            vmf = prototypes.vmf_loss(projected, labels, kappa=vmf_kappa)
            loss = loss + vmf_weight * vmf / gradient_accumulation_steps

        repulse = zero
        if prototypes is not None and repulse_weight > 0:
            repulse = prototypes.repulse_loss()
            loss = loss + repulse_weight * repulse / gradient_accumulation_steps

        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(proj_head.parameters(), max_norm=2.0)
            if prototypes is not None:
                torch.nn.utils.clip_grad_norm_([prototypes.prototypes], max_norm=2.0)
            optimizer.step()
            if prototypes is not None:
                # I prototipi restano sulla sfera unitaria: il gradiente li
                # muove nella direzione giusta, ma senza questa proiezione la
                # loro norma cresce (weight decay 0) e il passo efficace sulla
                # direzione si riduce a ogni epoca.
                with torch.no_grad():
                    prototypes.prototypes.data.copy_(F.normalize(prototypes.prototypes.data, dim=1))
            optimizer.zero_grad(set_to_none=True)

        n = inputs.size(0)
        running["loss"] += loss.detach() * gradient_accumulation_steps * n
        running["supcon"] += supcon.detach() * n
        running["vmf"] += vmf.detach() * n
        running["repulse"] += repulse.detach() * n
        num_samples += n

    return {key: float(value.item()) / max(num_samples, 1) for key, value in running.items()}


def _projected_embeddings(
    model: ClassifierModel,
    proj_head: ProjectionHead,
    data_loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalised projection-space features and labels for a whole loader."""
    was_training = model.training
    model.eval()
    proj_head.eval()
    feats, labels = [], []
    with torch.no_grad():
        for inputs, targets, _ in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            projected = proj_head(model.get_embeddings(inputs))
            feats.append(F.normalize(projected, dim=1).cpu())
            labels.append(targets.cpu())
    if was_training:
        model.train()
        proj_head.train()
    return torch.cat(feats), torch.cat(labels)


def _prototype_probe(
    prototypes,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    """Nearest-prototype accuracy/F1 in projection space, using the learned prototypes."""
    with torch.no_grad():
        sims = prototypes(features.to(prototypes.prototypes.device))
        preds = sims.argmax(dim=1).cpu()
    return (
        float(accuracy_score(labels.numpy(), preds.numpy())),
        float(f1_score(labels.numpy(), preds.numpy(), average="macro")),
    )


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


def _init_prototypes_from_loader(
    model: ClassifierModel,
    proj_head: ProjectionHead,
    loader,
    prototypes,
    device: torch.device,
) -> float:
    """Init dei prototipi sui centroidi di classe (fine del warm-up)."""
    feats, labels = _projected_embeddings(model, proj_head, loader, device)
    return prototypes.init_from_features(feats, labels)


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
    warmup_epochs: int = 20,
    vmf_kappa: float = 10.0,
    ramp_epochs: int = 20,
    repulse_mode: str = "squared",
    repulse_threshold: float = 0.3,
) -> tuple[ClassifierModel, dict]:
    """Phase 1: train backbone + projection head with SupCon loss.

    With ``use_prototypes=True`` the phase is split in two:

    1. **warm-up** (``warmup_epochs``): plain SupCon. The classes find their
       shape before any geometric constraint is applied.
    2. **joint phase**: the prototypes are initialised on the class centroids
       of the projected features, then the loss becomes
       ``SupCon + w(e) * vMF + repulse_weight * L_rep``, where ``w(e)`` grows
       linearly from 0 to 1 over ``ramp_epochs`` epochs. The vMF term is a
       hard assignment (each sample to its own prototype), so the prototypes
       are pulled towards the data instead of staying where they were
       initialised.

    The prototypes live in the projection space and are discarded with the
    projection head: they shape the backbone, which is what the open-set
    detectors use. They are returned in ``metrics["sphor"]`` for persistence.

    If pair_weights is provided, applies hard negative mining: specific class
    pairs receive higher loss weight to force separation of difficult pairs.

    Returns (model with best backbone weights, metrics dict).
    """

    model = model.to(device)
    proj_head = ProjectionHead(model.embedding_dim, proj_dim).to(device)

    criterion = SupConLoss(temperature=temperature, pair_weights=pair_weights)

    prototypes = None
    if use_prototypes:
        from app.core.losses import SphericalPrototypes

        prototypes = SphericalPrototypes(
            model.num_classes,
            proj_dim,
            repulse_mode=repulse_mode,
            repulse_threshold=repulse_threshold,
        ).to(device)
        criterion.prototypes = prototypes
        criterion._repulse_weight = repulse_weight
        logger.info(
            f"  Spherical prototypes: {model.num_classes} classi, dim {proj_dim}, "
            f"warm-up {warmup_epochs} ep, ramp {ramp_epochs} ep, kappa={vmf_kappa}, "
            f"repulse={repulse_weight} ({repulse_mode}, soglia {repulse_threshold})"
        )

    optimizer = _make_optimizer(model, proj_head, learning_rate, optimizer_type, weight_decay)
    if prototypes is not None:
        # I prototipi DEVONO stare nell'optimizer: senza questo il gradiente
        # della repulsione si accumula su .grad e non viene mai applicato, e il
        # metodo diventa un no-op (era il bug della I-C fino alla v0.6.0).
        # La guardia vive in tests/test_sphor.py.
        optimizer.add_param_group({"params": [prototypes.prototypes], "lr": learning_rate, "weight_decay": 0.0})
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    best_knn = 0.0
    best_backbone_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    start_epoch = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_knn_acc": [],
        "val_embedding_norm": [],
        "supcon_loss": [],
        "vmf_loss": [],
        "repulse_loss": [],
        "vmf_weight": [],
    }

    # Il warm-up non puo' mangiarsi l'intera fase: almeno un'epoca congiunta.
    warmup = max(0, min(int(warmup_epochs), max(num_epochs - 1, 0)))
    ramp = max(1, int(ramp_epochs))
    if use_prototypes and warmup != int(warmup_epochs):
        logger.warning(f"  warmup_epochs {warmup_epochs} ridotto a {warmup} (num_epochs={num_epochs})")
    protos_ready = False

    # Resume
    ckpt_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "last_ckpt_supcon.pt")

    if resume and ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        saved_groups = len(ckpt["optimizer_state_dict"].get("param_groups", []))
        if saved_groups != len(optimizer.param_groups):
            # SpHOR acceso/spento rispetto al run originale: `load_state_dict`
            # solleverebbe ValueError a meta' resume. Meglio ripartire puliti.
            logger.warning(
                f"Checkpoint con {saved_groups} param group, attesi {len(optimizer.param_groups)}: riparto da zero"
            )
        elif use_prototypes and "prototypes_state_dict" not in ckpt:
            # Checkpoint di un run precedente al metodo SpHOR vero: ripartire
            # da li' significherebbe mescolare due loss diverse.
            logger.warning("Checkpoint SupCon senza prototipi: riparto da zero (metodo cambiato)")
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            proj_head.load_state_dict(ckpt["proj_head_state_dict"])
            if prototypes is not None:
                prototypes.load_state_dict(ckpt["prototypes_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_knn = ckpt["best_knn"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            for key, value in ckpt["history"].items():
                history[key] = value
            # Un checkpoint vecchio non ha le chiavi nuove: le liste devono
            # restare lunghe quanto train_loss, altrimenti il fold JSON esce
            # con history di lunghezze diverse.
            done = len(history["train_loss"])
            for key in ("supcon_loss", "vmf_loss", "repulse_loss", "vmf_weight"):
                if len(history.get(key, [])) != done:
                    history[key] = [0.0] * done
            best_backbone_wts = copy.deepcopy(model.state_dict())
            protos_ready = prototypes is not None and start_epoch >= warmup
            logger.info(f"SupCon resumed from epoch {start_epoch}, best KNN: {best_knn:.4f}")

    for epoch in range(start_epoch, num_epochs):
        in_warmup = prototypes is not None and epoch < warmup
        if prototypes is not None and not protos_ready and not in_warmup:
            alignment = _init_prototypes_from_loader(model, proj_head, train_loader, prototypes, device)
            protos_ready = True
            logger.info(f"[SupCon] Prototipi inizializzati sui centroidi (cos medio {alignment:.4f})")

        vmf_weight = 0.0 if (prototypes is None or in_warmup) else min(1.0, (epoch - warmup + 1) / ramp)
        repulse_now = 0.0 if (prototypes is None or in_warmup) else repulse_weight

        logger.info(
            f"[SupCon] Epoch {epoch + 1}/{num_epochs}"
            + (" (warm-up, solo SupCon)" if in_warmup else f" (vMF w={vmf_weight:.2f}, repulse={repulse_now})")
        )

        parts = train_supcon_one_epoch(
            model,
            proj_head,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_accumulation_steps,
            vmf_weight=vmf_weight,
            vmf_kappa=vmf_kappa,
            repulse_weight=repulse_now,
        )
        train_loss = parts["loss"]

        knn_acc, avg_norm = validate_supcon(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_knn_acc"].append(knn_acc)
        history["val_embedding_norm"].append(avg_norm)
        history["supcon_loss"].append(parts["supcon"])
        history["vmf_loss"].append(parts["vmf"])
        history["repulse_loss"].append(parts["repulse"])
        history["vmf_weight"].append(vmf_weight)

        logger.info(
            f"[SupCon] Train Loss: {train_loss:.4f} (supcon {parts['supcon']:.4f}, "
            f"vmf {parts['vmf']:.4f}, rep {parts['repulse']:.4f}) | "
            f"Val KNN Acc: {knn_acc:.4f} | Emb Norm: {avg_norm:.4f}"
        )

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
            if prototypes is not None:
                ckpt_data["prototypes_state_dict"] = prototypes.state_dict()
            tmp_path = ckpt_path + ".tmp"
            torch.save(ckpt_data, tmp_path)
            os.replace(tmp_path, ckpt_path)

        # Ne' il warm-up ne' il ramp si tagliano. Se la KNN raggiunge il
        # massimo durante il warm-up (succede: e' la fase in cui il contrastivo
        # fa il lavoro grosso), con patience 15 la run si fermerebbe prima che
        # i termini SpHOR abbiano peso, e il modello salvato sarebbe quello del
        # warm-up: due varianti diverse darebbero lo stesso checkpoint byte per
        # byte e l'esperimento misurerebbe SupCon spacciato per SpHOR. E'
        # successo davvero: ablazione B e C, checkpoint identico.
        if prototypes is not None and epoch < warmup + ramp:
            epochs_no_improve = 0
        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"[SupCon] Early stopping at epoch {epoch + 1}")
            break

        scheduler.step(knn_acc)

    # Sonda SpHOR sul modello che ha appena finito la Fase 1, PRIMA di
    # ripristinare i pesi migliori: cosi' prototipi e backbone sono la coppia
    # coerente. Il checkpoint salvato resta quello con la KNN migliore, e la
    # sonda a centroidi in train_fold_supcon misura quello.
    sphor_metrics: dict = {}
    if prototypes is not None:
        feats, labels = _projected_embeddings(model, proj_head, val_loader, device)
        probe_acc, probe_f1 = _prototype_probe(prototypes, feats, labels)
        alignment = prototypes.alignment(feats, labels)
        sphor_metrics = {
            "warmup_epochs": warmup,
            "ramp_epochs": ramp,
            "kappa": float(vmf_kappa),
            "repulse_weight": float(repulse_weight),
            "repulse_mode": repulse_mode,
            "repulse_threshold": float(repulse_threshold),
            "alignment_val": alignment,
            "probe_accuracy": probe_acc,
            "probe_f1": probe_f1,
            "prototypes": prototypes.prototypes.detach().cpu().clone(),
        }
        logger.info(
            f"  SpHOR — prototipi: cos medio col centroide {alignment:.4f} | "
            f"probe (spazio proiezione) Acc {probe_acc:.4f}, F1 {probe_f1:.4f}"
        )

    model.load_state_dict(best_backbone_wts)
    logger.info(f"[SupCon] Phase 1 complete. Best KNN Acc: {best_knn:.4f}")

    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    del proj_head
    cleanup_gpu()

    return model, {
        "best_knn_acc": best_knn,
        "supcon_history": history,
        "sphor": sphor_metrics,
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
