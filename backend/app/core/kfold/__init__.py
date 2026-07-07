"""
K-Fold Cross-Validation con augmentation bilanciata per fold.
"""

import json
import logging
import os
import shutil
from datetime import datetime

import torch

from app.core.kfold.reporting import generate_report as generate_report
from app.core.kfold.testing import (
    aggregate_and_test,
    aggregate_results,
)
from app.core.kfold.training import (
    augment_fold_train,
    copy_fold_images,
    prepare_fold_data,
    split_raw_images,
    train_fold,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases (tests import these with underscore prefix)
_split_raw_images = split_raw_images
_copy_fold_images = copy_fold_images
_aggregate_results = aggregate_results


def kfold_experiment(
    raw_dataset_dir,
    test_dataset_dir,
    cropped_dataset_dir=None,
    model_variant="resnet18",
    num_folds=5,
    num_epochs=26,
    batch_size=8,
    learning_rate=0.001,
    seed=57,
    num_classes=6,
    class_names=None,
    output_dir="experiments/",
    kfold_dir=None,
    fold_index=None,
    aggregate_only=False,
    patience=0,
    gradient_accumulation_steps=1,
    loss_type="ce",
    proj_dim=128,
    temperature=0.07,
    ce_epochs=50,
    use_prototypes=False,
    repulse_weight=0.01,
    exclude_classes=None,
    pair_weights=None,
    full_finetune=False,
    optimizer_type=None,
    ncd_mode=False,
):
    """K-fold cross-validation con augmentation bilanciata per fold.
    optimizer_type: None = usa entry.optimizer dal registry, 'adam' = forza Adam per CE baseline.
    """
    from app.preprocessing.reproducibility import set_seed

    if class_names is None:
        from app.config import settings as _s

        class_names = _s.class_names

    if kfold_dir is None:
        kfold_dir = os.path.join(output_dir, f"kfold_{model_variant}")
    os.makedirs(kfold_dir, exist_ok=True)

    config_path = os.path.join(kfold_dir, "config.json")
    if not os.path.exists(config_path):
        from app.core.models import get_registry_entry as _get_entry

        _entry = _get_entry(model_variant)
        config = {
            "type": "kfold",
            "model_variant": model_variant,
            "num_folds": num_folds,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "num_classes": num_classes,
            "class_names": class_names,
            "raw_dataset_dir": raw_dataset_dir,
            "test_dataset_dir": test_dataset_dir,
            "patience": patience,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer": _entry.optimizer,
            "weight_decay": _entry.weight_decay,
            "effective_learning_rate": _entry.learning_rate or learning_rate,
            "loss_type": loss_type,
            "proj_dim": proj_dim if loss_type == "supcon" else None,
            "temperature": temperature if loss_type == "supcon" else None,
            "ce_epochs": ce_epochs if loss_type == "supcon" else None,
            "use_prototypes": use_prototypes if loss_type == "supcon" else None,
            "full_finetune": full_finetune if loss_type == "supcon" else None,
            "ncd_mode": ncd_mode if loss_type == "supcon" else None,
            "timestamp": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config salvato in: {config_path}")

    # Resume: CLI args sovrascrivono config.json salvato
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        for param, cli_val in [
            ("num_epochs", num_epochs),
            ("batch_size", batch_size),
            ("learning_rate", learning_rate),
            ("seed", seed),
            ("patience", patience),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
        ]:
            saved_val = saved.get(param)
            if saved_val is not None and saved_val != cli_val:
                logger.warning(f"Config {param}: {saved_val} → {cli_val} (CLI override)")
                saved[param] = cli_val
        num_epochs = saved.get("num_epochs", num_epochs)
        batch_size = saved.get("batch_size", batch_size)
        learning_rate = saved.get("learning_rate", learning_rate)
        seed = saved.get("seed", seed)
        patience = saved.get("patience", patience)
        gradient_accumulation_steps = saved.get("gradient_accumulation_steps", gradient_accumulation_steps)

    if aggregate_only:
        return aggregate_and_test(
            output_dir=kfold_dir,
            model_variant=model_variant,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            test_dataset_dir=test_dataset_dir,
            num_classes=num_classes,
            class_names=class_names,
        )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info(f"K-FOLD CROSS VALIDATION - {model_variant.upper()}")
    logger.info(f"Dataset: {raw_dataset_dir}")
    logger.info(f"Test: {test_dataset_dir}")
    logger.info(f"Folds: {num_folds}, Epochs: {num_epochs}, Batch size: {batch_size}, Patience: {patience}")
    if fold_index is not None:
        logger.info(f"MODALITA: fold singolo {fold_index}/{num_folds - 1}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)

    effective_dir = cropped_dataset_dir or raw_dataset_dir
    if cropped_dataset_dir:
        logger.info(f"Using pre-cropped dataset: {cropped_dataset_dir}")

    all_paths, all_labels, folds = split_raw_images(
        effective_dir, class_names, num_folds, seed, exclude_classes=exclude_classes
    )
    train_class_names = [c for c in class_names if c not in (exclude_classes or [])]

    if fold_index is not None:
        fold_indices = [fold_index]
    else:
        fold_indices = list(range(num_folds))

    base_temp_dir = None
    try:
        from app.core.models import get_registry_entry

        entry = get_registry_entry(model_variant)
        w, h = entry.input_size

        cache_tag = f"{w}x{h}"
        if exclude_classes:
            clean_names = [c.replace(" ", "_") for c in sorted(exclude_classes)]
            cache_tag += f"_no_{'_'.join(clean_names)}"
        base_temp_dir = os.path.join(os.path.dirname(effective_dir), f"temp_kfold_{cache_tag}")
        base_aug_cache = os.path.join(os.path.dirname(effective_dir), f"kfold_augmented_{cache_tag}")

        for fold_idx in fold_indices:
            fold_json_path = os.path.join(kfold_dir, f"fold_{fold_idx}.json")
            if os.path.exists(fold_json_path):
                # Validate partial JSON: try parsing it, skip if corrupted
                try:
                    with open(fold_json_path) as f:
                        json.load(f)
                except (json.JSONDecodeError, ValueError, OSError):
                    logger.warning(f"Fold {fold_idx} JSON corrotto, rifaccio...")
                    os.remove(fold_json_path)
                else:
                    logger.info(f"Fold {fold_idx} già completato (trovato {fold_json_path}). Salto...")
                    continue

            train_idx, val_idx = folds[fold_idx]

            logger.info(f"\n{'=' * 50}")
            logger.info(f"FOLD {fold_idx}/{num_folds - 1}")
            logger.info(f"Train: {len(train_idx)} immagini, Val: {len(val_idx)} immagini")
            logger.info(f"{'=' * 50}")

            fold_dir = os.path.join(base_temp_dir, f"fold_{fold_idx}")
            aug_cache_dir = os.path.join(base_aug_cache, f"fold_{fold_idx}")

            try:
                train_dir, val_dir = copy_fold_images(
                    all_paths, all_labels, train_class_names, train_idx, val_idx, fold_dir
                )
                augmented_dir = augment_fold_train(train_dir, train_class_names, w, h, cache_dir=aug_cache_dir)
                train_loader, val_loader = prepare_fold_data(
                    augmented_dir,
                    val_dir,
                    batch_size,
                    seed,
                    model_variant,
                    stratified=(loss_type == "supcon"),
                )
                checkpoint_path = os.path.join(kfold_dir, f"best_fold_{fold_idx}.pt")
                if loss_type == "supcon":
                    from app.core.kfold.training import train_fold_supcon

                    model, fold_metrics = train_fold_supcon(
                        model_variant,
                        num_classes,
                        train_loader,
                        val_loader,
                        num_epochs,
                        learning_rate,
                        device,
                        checkpoint_path,
                        proj_dim=proj_dim,
                        temperature=temperature,
                        ce_epochs=ce_epochs,
                        ce_patience=patience,
                        patience=patience,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        resume=True,
                        optimizer_type=optimizer_type,
                        use_prototypes=use_prototypes,
                        repulse_weight=repulse_weight,
                        pair_weights=pair_weights,
                        full_finetune=full_finetune,
                        ncd_mode=ncd_mode,
                    )
                else:
                    model, fold_metrics = train_fold(
                        model_variant,
                        num_classes,
                        train_loader,
                        val_loader,
                        num_epochs,
                        learning_rate,
                        device,
                        checkpoint_path,
                        patience=patience,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        resume=True,
                        optimizer_type=optimizer_type,
                    )

                fold_metrics["fold"] = fold_idx
                fold_metrics["batch_size"] = batch_size
                fold_metrics["train_size"] = len(train_idx)
                fold_metrics["val_size"] = len(val_idx)

                # GC: free model and loaders between folds
                del model
                del train_loader, val_loader
                from app.core.training import cleanup_gpu

                cleanup_gpu()

                fold_json_path = os.path.join(kfold_dir, f"fold_{fold_idx}.json")
                tmp_path = fold_json_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(fold_metrics, f, indent=2, default=str)
                os.replace(tmp_path, fold_json_path)
                logger.info(f"Fold {fold_idx} salvato in: {fold_json_path}")

                for ckpt in ("last_ckpt.pt", "last_ckpt_supcon.pt", "last_ckpt.pt.tmp", "last_ckpt_supcon.pt.tmp"):
                    path = os.path.join(kfold_dir, ckpt)
                    if os.path.exists(path):
                        os.remove(path)

            finally:
                if os.path.exists(fold_dir):
                    shutil.rmtree(fold_dir, ignore_errors=True)

    finally:
        if base_temp_dir and os.path.exists(base_temp_dir):
            shutil.rmtree(base_temp_dir, ignore_errors=True)

    if fold_index is not None:
        logger.info(f"\nFold {fold_index} completato.")
        logger.info(f"Per aggregare: --cross-validate --exp 6 --model {model_variant} --aggregate-only")
        return {"fold_completed": fold_index, "fold_metrics": fold_metrics}

    return aggregate_and_test(
        output_dir=kfold_dir,
        model_variant=model_variant,
        num_folds=num_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        test_dataset_dir=test_dataset_dir,
        num_classes=num_classes,
        class_names=class_names,
    )
