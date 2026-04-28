# app/core/kfold/training.py
"""Training functions for k-fold cross-validation."""
import json
import logging
import os
import shutil

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

import torch

logger = logging.getLogger(__name__)




def split_raw_images(raw_dataset_dir, class_names, num_folds, seed):
    """
    Divide le immagini originali in fold con StratifiedKFold.

    Returns:
        tuple: (all_paths, all_labels, folds) dove folds e' lista di (train_idx, val_idx)
    """
    all_paths = []
    all_labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(raw_dataset_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Classe {class_name} non trovata in {raw_dataset_dir}")
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(class_idx)

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(all_paths, all_labels))

    logger.info(f"Dataset: {len(all_paths)} immagini, {len(class_names)} classi")
    for cls_idx, cls_name in enumerate(class_names):
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
                    current_sizes[cn] = len([f for f in os.listdir(cp) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
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
            num_files = len([f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            class_sizes[class_name] = num_files
        else:
            class_sizes[class_name] = 0

    non_zero_sizes = [s for s in class_sizes.values() if s > 0]
    min_size = min(non_zero_sizes) if non_zero_sizes else 0
    target_per_class = min_size * len(aug_functions)

    logger.info(f"  Augmentation target: {target_per_class} immagini/classe "
                f"(min_class={min_size}, {len(aug_functions)} augmentation)")

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


def prepare_fold_data(augmented_train_dir, val_dir, batch_size, seed, model_variant):
    """
    Crea DataLoaders per un fold.

    Returns:
        tuple: (train_loader, val_loader)
    """
    from app.core.models import create_transform
    from app.preprocessing.dataset_tool import ImageFolderWithName
    from app.preprocessing.reproducibility import seed_worker
    from torch.utils.data import DataLoader

    transform = create_transform(model_variant)

    train_dataset = ImageFolderWithName(augmented_train_dir, transform=transform)
    val_dataset = ImageFolderWithName(val_dir, transform=transform)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
    )

    logger.info(f"  Train: {len(train_dataset)} immagini ({len(train_dataset.classes)} classi)")
    logger.info(f"  Val: {len(val_dataset)} immagini ({len(val_dataset.classes)} classi)")

    return train_loader, val_loader


def train_fold(model_variant, num_classes, train_loader, val_loader,
               num_epochs, learning_rate, device, checkpoint_path, patience=0,
               gradient_accumulation_steps=1, resume=False, overfit_threshold=0,
               optimizer_type="adam", weight_decay=0.0):
    """
    Train di un singolo fold. Riutilizza train() da app.core.training.

    Supports intra-fold resume: if resume=True and a last_ckpt.pt exists in the
    checkpoint directory, training continues from the saved epoch.

    Returns:
        tuple: (modello, fold_metrics_dict)
    """
    from app.core.models import create_model, get_registry_entry
    from app.core.training import train, cleanup_gpu

    ckpt_dir = os.path.dirname(checkpoint_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    model = create_model(model_variant, num_classes=num_classes, pretrained=True)

    # Use architecture-specific optimizer settings from registry
    entry = get_registry_entry(model_variant)
    opt_type = optimizer_type or entry.optimizer
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
        overfit_threshold=overfit_threshold,
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
