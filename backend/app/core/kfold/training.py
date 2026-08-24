# app/core/kfold/training.py
"""Training functions for k-fold cross-validation."""

import json
import logging
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


def split_raw_images(raw_dataset_dir, class_names, num_folds, seed, exclude_classes=None):
    """
    Divide le immagini originali in fold con StratifiedKFold.

    exclude_classes: classi da escludere dal training (es. lasciate come novel per NCD).

    Returns:
        tuple: (all_paths, all_labels, folds) dove folds e' lista di (train_idx, val_idx)
    """
    all_paths = []
    all_labels = []
    exclude = set(exclude_classes or [])

    active_names = [c for c in class_names if c not in exclude]
    for class_idx, class_name in enumerate(active_names):
        class_dir = os.path.join(raw_dataset_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Classe {class_name} non trovata in {raw_dataset_dir}")
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(class_idx)

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(all_paths, all_labels))

    logger.info(f"Dataset: {len(all_paths)} immagini, {len(active_names)} classi")
    for cls_idx, cls_name in enumerate(active_names):
        count = int((all_labels == cls_idx).sum())
        logger.info(f"  {cls_name}: {count} immagini")

    return all_paths, all_labels, folds


def copy_fold_images(all_paths, all_labels, class_names, train_idx, val_idx, fold_dir):
    """
    Copia le immagini originali nelle directory del fold.
    Anti-leakage: le immagini sono fisicamente separate prima dell'augmentation.

    Returns:
        tuple: (train_dir, val_dir)
    """
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")

    # Clean up any leftover files from a crashed previous run
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    # Crea directory per classe
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Copia immagini train
    for idx in train_idx:
        class_name = class_names[all_labels[idx]]
        dst = os.path.join(train_dir, class_name, os.path.basename(all_paths[idx]))
        try:
            shutil.copy(all_paths[idx], dst)
        except OSError as e:
            logger.warning(f"Skip copia {all_paths[idx]}: {e}")

    # Copia immagini val
    for idx in val_idx:
        class_name = class_names[all_labels[idx]]
        dst = os.path.join(val_dir, class_name, os.path.basename(all_paths[idx]))
        try:
            shutil.copy(all_paths[idx], dst)
        except OSError as e:
            logger.warning(f"Skip copia {all_paths[idx]}: {e}")

    return train_dir, val_dir


def augment_fold_train(fold_train_dir, class_names, width, height, cache_dir=None):
    """
    Augmentation bilanciata sulla directory train del fold.
    Usa augmentDataPathBalanced per classe con target calcolato sul fold corrente.

    Se cache_dir e' specificato, salva l'augmented in quella directory permanente
    e la riutilizza se esiste gia' (l'augmentation dipende solo dal dataset, non dal modello).

    Returns:
        str: percorso della directory augmentata
    """
    from app.preprocessing.augment import get_augmentation_functions
    from app.preprocessing.dataset_tool import augmentDataPathBalanced

    aug_functions = get_augmentation_functions()

    # Se abbiamo una cache dir, validiamo il manifest prima del riuso
    if cache_dir and os.path.exists(cache_dir):
        manifest_path = os.path.join(cache_dir, "_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Validate: same class names and same number of source images
            current_sizes = {}
            for cn in class_names:
                cp = os.path.join(fold_train_dir, cn)
                if os.path.exists(cp):
                    current_sizes[cn] = len(
                        [f for f in os.listdir(cp) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                    )
                else:
                    current_sizes[cn] = 0
            if manifest.get("source_sizes") == current_sizes and manifest.get("input_size") == [height, width]:
                logger.info(f"  Cache augmentation valida: {cache_dir}")
                return cache_dir
            else:
                logger.warning("  Cache augmentation stale (source_sizes changed), rigenero")
                shutil.rmtree(cache_dir)
        else:
            logger.warning("  Cache senza manifest, rigenero")
            shutil.rmtree(cache_dir)

    augmented_dir = cache_dir if cache_dir else fold_train_dir + "_augmented"

    # Calcola il target bilanciato per questo fold
    class_sizes = {}
    for class_name in class_names:
        class_path = os.path.join(fold_train_dir, class_name)
        if os.path.exists(class_path):
            num_files = len([f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
            class_sizes[class_name] = num_files
        else:
            class_sizes[class_name] = 0

    non_zero_sizes = [s for s in class_sizes.values() if s > 0]
    min_size = min(non_zero_sizes) if non_zero_sizes else 0
    target_per_class = min_size * len(aug_functions)

    logger.info(
        f"  Augmentation target: {target_per_class} immagini/classe "
        f"(min_class={min_size}, {len(aug_functions)} augmentation)"
    )

    # Augmenta ogni classe al target
    for class_name in class_names:
        input_path = os.path.join(fold_train_dir, class_name)
        output_path = os.path.join(augmented_dir, class_name)

        os.makedirs(output_path, exist_ok=True)

        if not os.path.exists(input_path) or class_sizes.get(class_name, 0) == 0:
            continue

        augmentDataPathBalanced(
            data_path=input_path,
            output_path=output_path,
            target_images=target_per_class,
            width=width,
            height=height,
            augmentation_functions=aug_functions,
            current_count=class_sizes[class_name],
        )

    # Salva manifest per validazione cache futura
    if cache_dir:
        manifest = {"source_sizes": class_sizes, "target_per_class": target_per_class, "input_size": [height, width]}
        with open(os.path.join(augmented_dir, "_manifest.json"), "w") as f:
            json.dump(manifest, f)

    return augmented_dir


def prepare_fold_data(augmented_train_dir, val_dir, batch_size, seed, model_variant, stratified=False):
    """
    Crea DataLoaders per un fold.

    Args:
        stratified: Se True, usa StratifiedBatchSampler per garantire ≥2 campioni/classe/batch.
                    Utile per SupConLoss (evita batch degeneri con gradiente zero).

    Returns:
        tuple: (train_loader, val_loader)
    """
    from app.core.models import create_transform
    from app.preprocessing.dataset_tool import ImageFolderWithName
    from app.preprocessing.reproducibility import seed_worker

    transform = create_transform(model_variant)

    train_dataset = ImageFolderWithName(augmented_train_dir, transform=transform)
    val_dataset = ImageFolderWithName(val_dir, transform=transform)

    if stratified:
        from app.core.kfold.sampler import StratifiedBatchSampler

        sampler = StratifiedBatchSampler(train_dataset.targets, batch_size, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="spawn",
            worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="spawn",
            worker_init_fn=seed_worker,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context="spawn",
        worker_init_fn=seed_worker,
    )

    logger.info(f"  Train: {len(train_dataset)} immagini ({len(train_dataset.classes)} classi)")
    logger.info(f"  Val: {len(val_dataset)} immagini ({len(val_dataset.classes)} classi)")

    return train_loader, val_loader


def train_fold(
    model_variant,
    num_classes,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    checkpoint_path,
    patience=0,
    gradient_accumulation_steps=1,
    resume=False,
    optimizer_type=None,
    weight_decay=0.0,
):
    """
    Train di un singolo fold. Riutilizza train() da app.core.training.

    Supports intra-fold resume: if resume=True and a last_ckpt.pt exists in the
    checkpoint directory, training continues from the saved epoch.

    Returns:
        tuple: (modello, fold_metrics_dict)
    """
    from app.core.models import create_model, get_registry_entry
    from app.core.training import train

    ckpt_dir = os.path.dirname(checkpoint_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    model = create_model(model_variant, num_classes=num_classes, pretrained=True)

    # Use architecture-specific optimizer settings from registry
    entry = get_registry_entry(model_variant)
    opt_type = entry.optimizer if optimizer_type is None else optimizer_type
    wd = weight_decay if weight_decay > 0 else entry.weight_decay
    lr = entry.learning_rate if entry.learning_rate else learning_rate

    if lr != learning_rate:
        logger.info(f"  Registry LR override: {learning_rate} -> {lr} ({model_variant})")

    resume_from = None
    if resume:
        candidate = os.path.join(ckpt_dir, "last_ckpt.pt")
        if os.path.exists(candidate):
            resume_from = candidate
            logger.info(f"  Resuming fold from checkpoint: {candidate}")

    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=lr,
        device=device,
        num_classes=num_classes,
        checkpoint_dir=ckpt_dir,
        patience=patience,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resume_from=resume_from,
        optimizer_type=opt_type,
        weight_decay=wd,
    )

    # Salva pesi migliori
    model.save_weights(checkpoint_path)

    best_f1 = max(history["val_f1"])
    best_epoch = history["val_f1"].index(best_f1) + 1
    best_acc = history["val_accuracy"][best_epoch - 1]

    fold_metrics = {
        "best_f1": float(best_f1),
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "history": {
            "train_loss": [float(x) for x in history["train_loss"]],
            "val_loss": [float(x) for x in history["val_loss"]],
            "val_accuracy": [float(x) for x in history["val_accuracy"]],
            "val_f1": [float(x) for x in history["val_f1"]],
        },
    }

    logger.info(f"  Fold completato: F1={best_f1:.4f}, Acc={best_acc:.4f}, Best epoch={best_epoch}")

    return model, fold_metrics


def _train_head_only(
    model,
    head_params,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    num_classes,
    model_variant=None,
    patience=0,
    gradient_accumulation_steps=1,
):
    """Train only the classifier head with backbone frozen. Lightweight loop for SupCon Phase 2."""
    from app.core.models.registry import get_registry_entry
    from app.core.torch_metrics import MulticlassAccuracy, MulticlassF1Score

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    entry = get_registry_entry(model_variant) if model_variant else None
    opt_type = entry.optimizer if entry else "adam"
    wd = entry.weight_decay if entry else 0.0
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(head_params, lr=learning_rate, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(head_params, lr=learning_rate)

    best_f1 = 0.0
    best_wts = {k: v.cpu() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs, labels, _) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * inputs.size(0)
            n += inputs.size(0)
        train_loss = running_loss / max(n, 1)

        model.eval()
        acc_m = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
        f1_m = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        val_loss = 0.0
        vb = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                out = model(inputs)
                val_loss += criterion(out, labels).item()
                vb += 1
                _, pred = torch.max(out, 1)
                acc_m.update(pred, labels)
                f1_m.update(pred, labels)
        val_loss /= max(vb, 1)
        accuracy = acc_m.compute().item()
        f1 = f1_m.compute().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(accuracy)
        history["val_f1"].append(f1)

        logger.info(
            f"[CE Head] Epoch {epoch + 1}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {accuracy:.4f} | F1: {f1:.4f}"
        )

        if f1 >= best_f1:
            best_f1 = f1
            best_wts = {k: v.cpu() for k, v in model.state_dict().items()}
            logger.info(f"[CE Head] New best F1: {f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"[CE Head] Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_wts)
    logger.info(f"[CE Head] Phase 2 complete. Best F1: {best_f1:.4f}")
    return model, history


def train_fold_supcon(
    model_variant,
    num_classes,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    checkpoint_path,
    proj_dim=128,
    temperature=0.07,
    ce_epochs=50,
    ce_patience=15,
    patience=0,
    gradient_accumulation_steps=1,
    resume=False,
    optimizer_type=None,
    weight_decay=0.0,
    use_prototypes=False,
    repulse_weight=0.05,
    pair_weights=None,
    full_finetune=False,
    ncd_mode=False,
):
    """Two-phase training for SupCon: Phase 1 (contrastive) + Phase 2 (CE fine-tune).

    Phase 1: backbone + projection head trained with SupCon loss.
    Phase 2: classifier head trained with CE loss (backbone optionally frozen).

    If full_finetune=True, the entire model is fine-tuned in Phase 2 with
    a lower learning rate for the backbone and a higher LR for the head.

    If pair_weights is provided, applies hard negative mining during Phase 1.

    Returns:
        tuple: (model, fold_metrics_dict with both phases)
    """
    from app.core.models import create_model, get_registry_entry
    from app.core.supcon import train_supcon

    ckpt_dir = os.path.dirname(checkpoint_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    if full_finetune:
        model = create_model(
            model_variant, num_classes=num_classes, pretrained=True, lora_r=32, normalize_embeddings=True
        )
        logger.info("  MGH-NCD mode: LoRA rank increased to 32 for richer embeddings")
    else:
        model = create_model(model_variant, num_classes=num_classes, pretrained=True, normalize_embeddings=True)
    entry = get_registry_entry(model_variant)
    opt_type = entry.optimizer if optimizer_type is None else optimizer_type
    wd = weight_decay if weight_decay > 0 else entry.weight_decay
    lr = entry.learning_rate if entry.learning_rate else learning_rate

    if lr != learning_rate:
        logger.info(f"  Registry LR override: {learning_rate} -> {lr} ({model_variant})")

    # --- Phase 1: SupCon with optional hard negative mining ---
    logger.info("  === Phase 1: SupCon Contrastive Learning ===")
    model, supcon_metrics = train_supcon(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=lr,
        device=device,
        checkpoint_dir=ckpt_dir,
        proj_dim=proj_dim,
        temperature=temperature,
        patience=patience or 15,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resume=resume,
        optimizer_type=opt_type,
        weight_decay=wd,
        use_prototypes=use_prototypes,
        repulse_weight=repulse_weight,
        pair_weights=pair_weights,
    )

    # --- Phase 2: CE fine-tuning (head only) ---
    # Both full_finetune and non-full_finetune branches use the same Phase 2 setup:
    # backbone frozen, head-only training. The full_finetune flag only affects
    # Phase 1 (LoRA rank: 32 vs 8) and NCD evaluation (ensemble OSR).
    if not ncd_mode:
        logger.info("  === Phase 2: CE Fine-tuning (head only) ===")
        for param in model.parameters():
            param.requires_grad = False
        model._unfreeze_head()

        head_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"  Trainable params (head only): {sum(p.numel() for p in head_params):,}")

        model, ce_history = _train_head_only(
            model=model,
            head_params=head_params,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=ce_epochs,
            learning_rate=lr,
            device=device,
            num_classes=num_classes,
            model_variant=model_variant,
            patience=ce_patience,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # Save best weights
        model.save_weights(checkpoint_path)

        best_f1 = max(ce_history["val_f1"])
        best_epoch = ce_history["val_f1"].index(best_f1) + 1
        best_acc = ce_history["val_accuracy"][best_epoch - 1]
    else:
        logger.info("  === NCD Mode: skipping Phase 2 (CE head) ===")
        supcon_history = supcon_metrics.get("supcon_history", {})
        supcon_val_knn = supcon_history.get("val_knn_acc", [])
        supcon_best_knn = supcon_metrics["best_knn_acc"]
        best_f1 = supcon_best_knn
        best_epoch = supcon_val_knn.index(supcon_best_knn) + 1 if supcon_val_knn else 0
        best_acc = 0.0
        ce_history = {"train_loss": [0.0], "val_loss": [0.0], "val_accuracy": [0.0], "val_f1": [0.0]}
        model.save_weights(checkpoint_path)

    # --- SpHOR prototype probe (if using prototypes) ---
    sphor_probe_accuracy = None
    sphor_probe_f1 = None

    if use_prototypes:
        logger.info("  === SpHOR Prototype Probe (nearest centroid) ===")
        model.eval()
        centroids, all_preds, all_labels, embed_dim = {}, [], [], None

        # Split val_loader: prima metà per centroidi, seconda per evaluation
        val_dataset = val_loader.dataset
        n = len(val_dataset)
        mid = n // 2
        centroid_loader = DataLoader(
            Subset(val_dataset, range(mid)),
            batch_size=val_loader.batch_size,
            shuffle=False,
        )
        eval_loader = DataLoader(
            Subset(val_dataset, range(mid, n)),
            batch_size=val_loader.batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for inputs, labels, _ in centroid_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                embeddings = model.get_embeddings(inputs)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                if embed_dim is None:
                    embed_dim = embeddings.shape[-1]

                for i in range(len(labels)):
                    lbl = labels[i].item()
                    centroids.setdefault(lbl, []).append(embeddings[i])

        if embed_dim is None:
            embed_dim = 768
        proto = torch.zeros(num_classes, embed_dim).to(device)
        for lbl in range(num_classes):
            if lbl in centroids:
                proto[lbl] = torch.stack(centroids[lbl]).mean(dim=0)
        proto = F.normalize(proto, p=2, dim=1)

        with torch.no_grad():
            for inputs, labels, _ in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                embeddings = model.get_embeddings(inputs)
                embeddings = F.normalize(embeddings, p=2, dim=1)

                sim = embeddings @ proto.T
                preds = sim.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        sphor_probe_accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
        sphor_probe_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
        logger.info(f"  SpHOR Probe — Acc: {sphor_probe_accuracy:.4f}, F1: {sphor_probe_f1:.4f}")

    fold_metrics = {
        "best_f1": float(best_f1),
        "best_accuracy": float(best_acc),
        "sphor_probe_accuracy": sphor_probe_accuracy,
        "sphor_probe_f1": sphor_probe_f1,
        "best_epoch": best_epoch,
        "final_train_loss": float(ce_history["train_loss"][-1]),
        "final_val_loss": float(ce_history["val_loss"][-1]),
        "supcon_phase": supcon_metrics,
        "history": {
            "train_loss": [float(x) for x in ce_history["train_loss"]],
            "val_loss": [float(x) for x in ce_history["val_loss"]],
            "val_accuracy": [float(x) for x in ce_history["val_accuracy"]],
            "val_f1": [float(x) for x in ce_history["val_f1"]],
        },
    }

    phase_label = "SupCon+NCD" if ncd_mode else "SupCon+CE"
    logger.info(
        f"  Fold completato ({phase_label}): F1={best_f1:.4f}, Acc={best_acc:.4f}, "
        f"Best epoch={best_epoch}, SupCon KNN={supcon_metrics['best_knn_acc']:.4f}"
    )

    return model, fold_metrics
