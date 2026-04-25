# app/cli/runner.py
"""
Core experiment execution logic: device setup, logging, status display, and
the main run_experiment() orchestrator.
"""
import json
import logging
import os
import torch
from datetime import datetime

from app.core.models import create_model, create_transform
from app.cli.definitions import (
    PROJECT_ROOT,
    DATASET_PATHS,
    ALL_CLASSES,
    EXPERIMENTS,
    DEFAULT_CONFIG,
)
from app.cli.reports import _generate_experiment_report


def setup_logging(log_dir="logs"):
    """Setup logging for experiments — console only. File logging is per-experiment."""
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Solo console handler — i file log vengono gestiti per-esperimento
    if not root_logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        root_logger.addHandler(console)

    return log_dir


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def show_status():
    """Show status of all experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT STATUS")
    print("=" * 70)

    experiments_dir = PROJECT_ROOT / "experiments"

    for exp_id, exp_config in EXPERIMENTS.items():
        exp_dir = experiments_dir / exp_config["name"]
        # Cerca il done file nella directory base o in qualsiasi subdir con tag
        done_file = exp_dir / f".exp_{exp_id}_done"
        tag_done_files = list(experiments_dir.glob(f"{exp_config['name']}_*/.exp_{exp_id}_done"))

        if done_file.exists() or tag_done_files:
            status = "✓ COMPLETED"
        elif exp_dir.exists() or list(experiments_dir.glob(f"{exp_config['name']}_*")):
            status = "⏳ IN PROGRESS"
        else:
            status = "○ NOT STARTED"

        print(f"Exp{exp_id}: {exp_config['name']:40} [{status}]")
        print(f"       {exp_config['description']}")
        print(f"       Train: {exp_config['train_dataset']:15} → Test: {exp_config['test_dataset']}")
        print()

    print("=" * 70 + "\n")


def run_experiment(
    exp_id: int,
    phases: list,
    model_variant: str = "resnet18",
    epochs: int = 26,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    pretrained: bool = True,
    num_folds: int = 5,
    fold_index: int = None,
    aggregate_only: bool = False,
    patience: int = 0,
    tag: str = None,
):
    """
    Run a single experiment.

    Args:
        exp_id: Experiment ID (1-5)
        phases: List of phases to run ['train', 'test', 'cross-validate']
        model_variant: Model variant (e.g. 'resnet18', 'resnet50', 'convnext_tiny', 'dinov2_small')
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        pretrained: Use pretrained weights
        num_folds: Number of folds for cross-validation
    """
    from torch.utils.data import DataLoader
    from app.core.training import train, cross_validate
    from app.core.evaluation import evaluate
    from app.preprocessing.dataset_tool import getDatasetFromFile

    exp = EXPERIMENTS[exp_id]
    device = get_device()

    # Per-experiment log file (dentro experiments/)
    results_dir = PROJECT_ROOT / "experiments" / (f"{exp['name']}_{tag}" if tag else exp["name"])
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_log_name = f"{exp['name']}_{tag}_{model_variant}.log" if tag else f"{exp['name']}_{model_variant}.log"
    exp_log = str(results_dir / exp_log_name)
    exp_handler = logging.FileHandler(exp_log)
    exp_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(exp_handler)

    # Force flush dopo ogni log
    orig_emit = exp_handler.emit
    def flushing_emit(record):
        orig_emit(record)
        exp_handler.flush()
    exp_handler.emit = flushing_emit

    logging.info("=" * 70)
    logging.info(f"EXPERIMENT {exp_id}: {exp['name']}")
    logging.info(f"Model: {model_variant.upper()}")
    logging.info(f"Description: {exp['description']}")
    logging.info(f"Log: {exp_log}")
    logging.info("=" * 70)

    try:
        # --- Experiment body ---

        model_dir = PROJECT_ROOT / "models" / f"train_{exp['train_dataset']}_{model_variant}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"

        # experiments_dir already created above for log file

        # Create experiment tracking directory
        exp_dir = PROJECT_ROOT / "experiments" / exp["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        train_path = DATASET_PATHS[exp["train_dataset"]]["train"]
        val_path = DATASET_PATHS[exp["train_dataset"]]["val"]
        test_path = DATASET_PATHS[exp["test_dataset"]]["test"]

        logging.info(f"Train: {train_path}")
        logging.info(f"Val: {val_path}")
        logging.info(f"Test: {test_path}")

        # Run phases
        if "cross-validate" in phases:
            logging.info("\n--- PHASE: K-Fold Cross Validation ---")
            logging.info(f"Folds: {num_folds}")

            if exp_id == 4:
                # K-fold con augmentation bilanciata per fold (no data leakage)
                from app.core.kfold import kfold_experiment

                raw_dir = str(PROJECT_ROOT / DATASET_PATHS["combined"]["raw"])
                test_dir = str(PROJECT_ROOT / DATASET_PATHS["internet_cropped"]["raw"])

                kfold_experiment(
                    raw_dataset_dir=raw_dir,
                    test_dataset_dir=test_dir,
                    model_variant=model_variant,
                    num_folds=num_folds,
                    num_epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=DEFAULT_CONFIG["seed"],
                    num_classes=exp["num_classes"],
                    class_names=ALL_CLASSES,
                    output_dir=str(results_dir),
                    kfold_dir=str(results_dir),
                    fold_index=fold_index,
                    aggregate_only=aggregate_only,
                    patience=patience,
                )
            else:
                # Cross-validation originale su dataset pre-augmentato
                from app.core.models import get_registry_entry

                train_dataset = getDatasetFromFile(train_path)

                config = get_registry_entry(model_variant).config_cls(
                    variant=model_variant,
                    num_classes=exp["num_classes"],
                    pretrained=pretrained,
                )

                cv_results = cross_validate(
                    model_config=config,
                    dataset=train_dataset,
                    num_folds=num_folds,
                    num_epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    device=device,
                    num_classes=exp["num_classes"],
                    patience=patience,
                )

                # Save CV results
                cv_path = results_dir / f"cross_validation_{model_variant}.json"
                with open(cv_path, "w") as f:
                    json.dump({
                        "mean_f1": float(cv_results["mean_f1"]),
                        "std_f1": float(cv_results["std_f1"]),
                        "mean_accuracy": float(cv_results["mean_accuracy"]),
                        "std_accuracy": float(cv_results["std_accuracy"]),
                        "num_folds": num_folds,
                        "model_variant": model_variant,
                    }, f, indent=2)
                logging.info(f"CV results saved to: {cv_path}")

        if "train" in phases:
            logging.info("\n--- PHASE: Training ---")

            train_dataset = getDatasetFromFile(train_path)
            val_dataset = getDatasetFromFile(val_path)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = create_model(model_variant, num_classes=exp["num_classes"], pretrained=pretrained)

            model, history = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                num_classes=exp["num_classes"],
                patience=patience,
            )

            # Save model
            model.save_weights(str(model_path))
            logging.info(f"Model saved to: {model_path}")

        if "test" in phases:
            logging.info("\n--- PHASE: Testing ---")

            # Load model if not already loaded
            if "train" not in phases:
                model = create_model(model_variant, num_classes=exp["num_classes"], pretrained=pretrained)
                model.load_weights(str(model_path))

            model = model.to(device)

            # Load test dataset
            test_dataset = getDatasetFromFile(test_path)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Evaluate
            results = evaluate(model, test_loader, exp["test_classes"], device)

            logging.info(f"Test Accuracy: {results['accuracy']:.4f}")
            logging.info(f"Test F1 (Macro): {results['f1_macro']:.4f}")
            logging.info(f"Test F1 (Micro): {results['f1_micro']:.4f}")

            # Save results JSON
            results_path = results_dir / f"test_results_{model_variant}.json"
            with open(results_path, "w") as f:
                json.dump({
                    "accuracy": float(results["accuracy"]),
                    "f1_macro": float(results["f1_macro"]),
                    "f1_micro": float(results["f1_micro"]),
                    "model_variant": model_variant,
                }, f, indent=2)
            logging.info(f"Results saved to: {results_path}")

            # Generate text report
            report_path = results_dir / f"REPORT_{model_variant}.txt"
            _generate_experiment_report(
                results=results,
                model_variant=model_variant,
                exp=exp,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_path=report_path,
            )
            logging.info(f"Report saved to: {report_path}")

        # --- PHASE: Explainability ---
        if "explain" in phases:
            logging.info("\n--- PHASE: Explainability ---")
            skip_explain = False
            try:
                import matplotlib
                matplotlib.use('Agg')
                from app.explainability.visualization import generate_output_images
                from app.preprocessing.dataset_tool import ImageFolderWithName

                # Determine model path based on experiment type
                best_model_path = None
                if exp_id == 4:
                    # K-fold experiment: use best fold model
                    kfold_exp_dir = str(results_dir)
                    kfold_results_path = os.path.join(kfold_exp_dir, "results.json")
                    if not os.path.exists(kfold_results_path):
                        logging.warning("--explain requires completed k-fold results. Run with --cross-validate first")
                        skip_explain = True
                    else:
                        with open(kfold_results_path) as f:
                            kfold_data = json.load(f)
                        best_fold = kfold_data.get("best_fold", 0)
                        best_model_path = os.path.join(kfold_exp_dir, f"best_fold_{best_fold}.pt")
                else:
                    # Standard experiment: use last trained model
                    best_model_path = str(model_dir / "model.pt")

                if not skip_explain and not os.path.exists(best_model_path):
                    logging.warning(f"Model not found: {best_model_path}")
                    skip_explain = True

                if not skip_explain:
                    model = create_model(model_variant, num_classes=exp["num_classes"], pretrained=False)
                    model.load_weights(best_model_path)
                    model = model.to(device)

                    transform = create_transform(model_variant)
                    test_raw_dir = str(PROJECT_ROOT / DATASET_PATHS[exp["test_dataset"]]["raw"])
                    test_dataset = ImageFolderWithName(test_raw_dir, transform=transform)
                    explain_dir = str(results_dir / "explainability")
                    os.makedirs(explain_dir, exist_ok=True)
                    generate_output_images(
                        dataset=test_dataset,
                        model=model,
                        device=device,
                        class_names=ALL_CLASSES,
                        output_dir=str(explain_dir),
                        sliding_window_size=8,
                        stride=4,
                    )
                    logging.info(f"Explainability images saved to {explain_dir}")
            except Exception as e:
                logging.error(f"Explainability failed: {e}")
                import traceback
                traceback.print_exc()

        # Mark as completed
        done_file = results_dir / f".exp_{exp_id}_done"
        done_file.write_text(f"Completed at {datetime.now().isoformat()}\nModel: {model_variant}\n")

        logging.info(f"\n✓ EXPERIMENT {exp_id} COMPLETED")
        return True

    finally:
        # Remove per-experiment log handler
        logging.getLogger().removeHandler(exp_handler)
        exp_handler.close()
