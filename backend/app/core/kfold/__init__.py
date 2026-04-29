"""
K-Fold Cross-Validation con augmentation bilanciata per fold.
"""
import json
import logging
import os
from datetime import datetime

import torch

from app.core.kfold.training import (
    split_raw_images,
    copy_fold_images,
    augment_fold_train,
    prepare_fold_data,
    train_fold,
)
from app.core.kfold.testing import (
    aggregate_results,
    final_test,
    aggregate_and_test,
)
from app.core.kfold.reporting import generate_report

logger = logging.getLogger(__name__)

# Backward-compatible aliases (tests import these with underscore prefix)
_split_raw_images = split_raw_images
_copy_fold_images = copy_fold_images
_aggregate_results = aggregate_results




def kfold_experiment(
    raw_dataset_dir,
    test_dataset_dir,
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
    overfit_threshold=0,
):
    """K-fold cross-validation con augmentation bilanciata per fold."""
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
            "type": "kfold", "model_variant": model_variant, "num_folds": num_folds,
            "num_epochs": num_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "seed": seed, "num_classes": num_classes, "class_names": class_names,
            "raw_dataset_dir": raw_dataset_dir, "test_dataset_dir": test_dataset_dir,
            "patience": patience,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "overfit_threshold": overfit_threshold,
            "optimizer": _entry.optimizer, "weight_decay": _entry.weight_decay,
            "effective_learning_rate": _entry.learning_rate or learning_rate,
            "timestamp": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config salvato in: {config_path}")

    # Resume: leggi parametri dal config salvato se diverso da CLI
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        resume_params = {
            "num_epochs": num_epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "seed": seed, "patience": patience,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "overfit_threshold": overfit_threshold,
        }
        for param, cli_val in resume_params.items():
            if param in saved and saved[param] != cli_val:
                logger.warning(
                    f"Resume: {param} = {saved[param]} (da config.json, override di {cli_val} da CLI)"
                )
        num_epochs = saved.get("num_epochs", num_epochs)
        batch_size = saved.get("batch_size", batch_size)
        learning_rate = saved.get("learning_rate", learning_rate)
        seed = saved.get("seed", seed)
        patience = saved.get("patience", patience)
        gradient_accumulation_steps = saved.get("gradient_accumulation_steps", gradient_accumulation_steps)
        overfit_threshold = saved.get("overfit_threshold", overfit_threshold)

    if aggregate_only:
        return aggregate_and_test(
            output_dir=kfold_dir, model_variant=model_variant, num_folds=num_folds,
            num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
            seed=seed, test_dataset_dir=test_dataset_dir,
            num_classes=num_classes, class_names=class_names,
        )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info(f"K-FOLD CROSS VALIDATION - {model_variant.upper()}")
    logger.info(f"Dataset: {raw_dataset_dir}")
    logger.info(f"Test: {test_dataset_dir}")
    logger.info(f"Folds: {num_folds}, Epochs: {num_epochs}, Batch size: {batch_size}, Patience: {patience}")
    if fold_index is not None:
        logger.info(f"MODALITA: fold singolo {fold_index}/{num_folds}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)

    all_paths, all_labels, folds = split_raw_images(
        raw_dataset_dir, class_names, num_folds, seed
    )

    if fold_index is not None:
        fold_indices = [fold_index]
    else:
        fold_indices = list(range(num_folds))

    import shutil
    try:
        from app.core.models import get_registry_entry
        entry = get_registry_entry(model_variant)
        h, w = entry.input_size

        base_temp_dir = os.path.join(os.path.dirname(raw_dataset_dir), "temp_kfold")
        base_aug_cache = os.path.join(os.path.dirname(raw_dataset_dir), f"kfold_augmented_{w}x{h}")

        for fold_idx in fold_indices:
            fold_json_path = os.path.join(kfold_dir, f"fold_{fold_idx}.json")
            if os.path.exists(fold_json_path):
                logger.info(f"Fold {fold_idx} già completato (trovato {fold_json_path}). Salto...")
                continue

            train_idx, val_idx = folds[fold_idx]

            logger.info(f"\n{'=' * 50}")
            logger.info(f"FOLD {fold_idx}/{num_folds}")
            logger.info(f"Train: {len(train_idx)} immagini, Val: {len(val_idx)} immagini")
            logger.info(f"{'=' * 50}")

            fold_dir = os.path.join(base_temp_dir, f"fold_{fold_idx}")
            aug_cache_dir = os.path.join(base_aug_cache, f"fold_{fold_idx}")

            try:
                train_dir, val_dir = copy_fold_images(
                    all_paths, all_labels, class_names,
                    train_idx, val_idx, fold_dir
                )
                augmented_dir = augment_fold_train(
                    train_dir, class_names, w, h,
                    cache_dir=aug_cache_dir
                )
                train_loader, val_loader = prepare_fold_data(
                    augmented_dir, val_dir, batch_size, seed, model_variant
                )
                checkpoint_path = os.path.join(kfold_dir, f"best_fold_{fold_idx}.pt")
                model, fold_metrics = train_fold(
                    model_variant, num_classes, train_loader, val_loader,
                    num_epochs, learning_rate, device, checkpoint_path,
                    patience=patience,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    resume=True,
                    overfit_threshold=overfit_threshold,
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
                with open(fold_json_path, "w") as f:
                    json.dump(fold_metrics, f, indent=2, default=str)
                logger.info(f"Fold {fold_idx} salvato in: {fold_json_path}")

            finally:
                if os.path.exists(fold_dir):
                    shutil.rmtree(fold_dir, ignore_errors=True)

    finally:
        if os.path.exists(base_temp_dir):
            shutil.rmtree(base_temp_dir, ignore_errors=True)

    if fold_index is not None:
        logger.info(f"\nFold {fold_index} completato.")
        logger.info(f"Per aggregare: --cross-validate --exp 6 --model {model_variant} --aggregate-only")
        return {"fold_completed": fold_index, "fold_metrics": fold_metrics}

    return aggregate_and_test(
        output_dir=kfold_dir, model_variant=model_variant, num_folds=num_folds,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        seed=seed, test_dataset_dir=test_dataset_dir,
        num_classes=num_classes, class_names=class_names,
    )
