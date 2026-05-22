"""Utility functions for CLI: device detection, logging setup, status display."""

import logging
import os

import torch

from app.cli.definitions import EXPERIMENTS, PROJECT_ROOT


def setup_logging(log_dir: str | None = None, level: int = logging.INFO) -> str:
    """Configure logging: file + stderr."""
    log_dir = log_dir or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "cli.log")

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    root.addHandler(console)

    logging.info(f"Logging to {log_file}")
    return log_file


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def show_status():
    """Print experiment status table."""
    logging.getLogger().setLevel(logging.WARNING)
    print(experiment_status_table())

    from app.core.models import available_models
    models = available_models()
    if models:
        print(f"\n  Registered models: {', '.join(models)}")


def experiment_status_table() -> str:
    """Build ASCII table of experiment status."""
    lines = ["\n" + "=" * 60, "  EXPERIMENT STATUS", "=" * 60]
    for eid in sorted(EXPERIMENTS):
        config = EXPERIMENTS[eid]
        name = config.get("name", "?")
        train = config.get("train_dataset", "?")
        test = config.get("test_dataset", "?")
        description = config.get("description", "")
        lines.append(f"  Exp{eid}: {name}")
        lines.append(f"    Train: {train}")
        lines.append(f"    Test:  {test}")
        if description:
            lines.append(f"    {description}")
        lines.append("")
    return "\n".join(lines)


def run_experiment(
    exp_id: int,
    phases: list[str] | None = None,
    model_variant: str = "resnet18",
    epochs: int = 26,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    num_folds: int = 5,
    fold_index: int | None = None,
    aggregate_only: bool = False,
    patience: int = 15,
    tag: str = "",
    gradient_accumulation_steps: int = 1,
    proj_dim: int = 128,
    temperature: float = 0.07,
    ce_epochs: int = 50,
    use_prototypes: bool = False,
    repulse_weight: float = 0.05,
    pair_weights: dict | None = None,
    full_finetune: bool = False,
    osr_ensemble: bool = False,
):
    """Run a single experiment (k-fold cross-validation or single fold)."""
    import logging

    from app.cli.definitions import ALL_CLASSES, DATASET_PATHS, EXPERIMENTS, PROJECT_ROOT
    from app.core.kfold.testing import aggregate_results

    logger = logging.getLogger(__name__)
    config = EXPERIMENTS[exp_id]
    output_dir = os.path.join(PROJECT_ROOT, "experiments", f"{config['name']}_{model_variant}")
    os.makedirs(output_dir, exist_ok=True)

    # Per-experiment log file
    exp_log_name = f"{config['name']}_{model_variant}.log"
    exp_log = os.path.join(output_dir, exp_log_name)
    exp_handler = logging.FileHandler(exp_log)
    exp_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(exp_handler)
    # Force flush dopo ogni log
    orig_emit = exp_handler.emit
    def flushing_emit(record):
        orig_emit(record)
        exp_handler.flush()
    exp_handler.emit = flushing_emit

    logger.info(f"Log file: {exp_log}")

    is_supcon = config.get("loss_type") == "supcon"

    # Embedding extraction phase (standalone)
    if phases and "embeddings" in phases:
        logger.info(f"Extracting embeddings for {model_variant}...")
        from app.core.models import get_registry_entry
        from app.core.ncd.extract import run_embedding_extraction
        entry = get_registry_entry(model_variant)
        w, h = entry.input_size
        cropped_train = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / "train_clean")
        if not os.path.exists(cropped_train):
            logger.error(f"Dataset not found: {cropped_train}")
            return
        run_embedding_extraction(
            exp_dir=output_dir,
            model_variant=model_variant,
            num_classes=config["num_classes"],
            class_names=config.get("test_classes", ALL_CLASSES),
            data_dir=cropped_train,
            batch_size=batch_size,
        )
        return

    # NCD phase (standalone)
    if phases and "ncd" in phases:
        logger.info(f"Running NCD pipeline for {model_variant}...")
        from app.core.models import get_registry_entry
        from app.core.ncd.pipeline import run_ncd_pipeline
        entry = get_registry_entry(model_variant)
        w, h = entry.input_size
        cropped_train = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / "train_clean")
        cropped_test = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / "test_clean")
        if not os.path.exists(cropped_train):
            logger.error(f"Dataset not found: {cropped_train}")
            return
        run_ncd_pipeline(
            exp_dir=output_dir,
            model_variant=model_variant,
            num_classes=config["num_classes"],
            class_names=config.get("test_classes", ALL_CLASSES),
            exclude_classes=config.get("exclude_classes"),
            data_dir=cropped_train,
            batch_size=batch_size,
            run_osr=True,
            test_dir=cropped_test if os.path.exists(cropped_test) else None,
            osr_method="ensemble" if osr_ensemble else "openmax",
        )
        return

    if aggregate_only:
        logger.info(f"Aggregating results for {model_variant}...")
        import json as _json

        from app.cli.definitions import ALL_CLASSES, DATASET_PATHS
        from app.core.kfold.reporting import generate_report as gen_report
        from app.core.kfold.testing import final_test

        # Read fold results
        fold_results = []
        for fi in range(num_folds):
            fp = os.path.join(output_dir, f"fold_{fi}.json")
            if os.path.exists(fp):
                with open(fp) as f:
                    fold_results.append(_json.load(f))

        if not fold_results:
            logger.error("No fold results found")
            return

        agg = aggregate_results(fold_results)

        # Run final test on best fold
        best_fold = agg.get("best_fold", 0)
        best_model_path = os.path.join(output_dir, f"best_fold_{best_fold}.pt")
        test_key = config["test_dataset"]
        test_dir = str(PROJECT_ROOT / DATASET_PATHS.get(test_key, {}).get("test", test_key))
        num_classes_val = config.get("num_classes", 6)
        class_names = config.get("test_classes", ALL_CLASSES)

        test_metrics = final_test(
            best_model_path, model_variant, num_classes_val, test_dir,
            class_names, batch_size, get_device()
        )

        # Save combined results
        results = {**agg, "test_metrics": test_metrics, "model_variant": model_variant}
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, "w") as f:
            _json.dump(results, f, indent=2, default=str)

        gen_report(results, model_variant, output_dir)

        # Upload to HF
        from app.core.kfold.testing import _upload_to_hf
        _upload_to_hf(output_dir, model_variant)
        return

    def _resolve_path(key: str) -> str:
        paths = DATASET_PATHS.get(config.get(key, ""), {})
        return str(PROJECT_ROOT / paths.get("train", paths.get("raw", paths.get("test", ""))))

    train_path = _resolve_path("train_dataset")
    test_path = _resolve_path("test_dataset")

    if is_supcon:
        from app.core.kfold import kfold_experiment
        from app.core.models import get_registry_entry
        entry = get_registry_entry(model_variant)
        w, h = entry.input_size
        cropped_train = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / "train_clean")
        cropped_test = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / "test_clean")
        if os.path.exists(cropped_train) and os.path.exists(cropped_test):
            raw_dir = cropped_train
            test_dir = cropped_test
            logger.info(f"Using pre-cropped dataset: {cropped_train}")
        else:
            raw_dir = train_path
            test_dir = test_path

        kfold_experiment(
            raw_dataset_dir=raw_dir,
            test_dataset_dir=test_dir,
            model_variant=model_variant,
            num_folds=num_folds,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=57,
            num_classes=config["num_classes"],
            class_names=config.get("test_classes", ALL_CLASSES),
            output_dir=output_dir,
            kfold_dir=output_dir,
            fold_index=fold_index,
            aggregate_only=False,
            patience=patience,
            gradient_accumulation_steps=gradient_accumulation_steps,
            loss_type="supcon",
            proj_dim=proj_dim,
            temperature=temperature,
            ce_epochs=ce_epochs,
            use_prototypes=use_prototypes,
            repulse_weight=repulse_weight,
            exclude_classes=config.get("exclude_classes"),
            pair_weights=pair_weights,
            full_finetune=full_finetune,
        )
    else:
        from app.core.kfold import kfold_experiment
        kfold_experiment(
            raw_dataset_dir=train_path,
            test_dataset_dir=test_path,
            model_variant=model_variant,
            num_folds=num_folds,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=57,
            num_classes=config["num_classes"],
            class_names=config.get("test_classes", ALL_CLASSES),
            output_dir=output_dir,
            kfold_dir=output_dir,
            fold_index=fold_index,
            aggregate_only=False,
            patience=patience,
            gradient_accumulation_steps=gradient_accumulation_steps,
            loss_type="ce",
            exclude_classes=config.get("exclude_classes"),
        )
