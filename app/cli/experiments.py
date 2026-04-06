# app/cli/experiments.py
"""
Orchestrator for running multiple classification experiments.

Each experiment has a specific train/test dataset combination.
Supports:
- Multiple model variants (ResNet18, ResNet50)
- K-fold cross validation
- Standard train/test split
"""
import argparse
import sys
import os
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Literal

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dataset paths
DATASET_PATHS = {
    "privato": {
        "raw": "dataset/raw/",
        "augmented": "dataset/augmented_privato/",
        "train": "dataset/train_privato.pt",
        "val": "dataset/val_privato.pt",
        "test": "dataset/test_privato.pt",
    },
    "internet_raw": {
        "raw": "dataset/internet_raw/",
        "test": "dataset/test_internet_raw.pt",
    },
    "internet_cropped": {
        "raw": "dataset/internet_cropped/",
        "test": "dataset/test_internet_cropped.pt",
    },
    "combined": {
        "raw": "dataset/combined/",
        "augmented": "dataset/augmented_combined/",
        "train": "dataset/train_combined.pt",
        "val": "dataset/val_combined.pt",
        "test": "dataset/test_combined.pt",
    }
}

# Class definitions
ALL_CLASSES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. majellensis",
    "O. sphegodes",
    "O. sphegodes_Palena"
]

INTERNET_CLASSES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. sphegodes"
]

# Experiment definitions
EXPERIMENTS = {
    1: {
        "name": "exp1_privato_privato",
        "train_dataset": "privato",
        "test_dataset": "privato",
        "num_classes": 6,
        "test_classes": ALL_CLASSES,
        "description": "Baseline: train and test on private dataset (6 classes)"
    },
    2: {
        "name": "exp2_privato_internet_raw",
        "train_dataset": "privato",
        "test_dataset": "internet_raw",
        "num_classes": 6,
        "test_classes": INTERNET_CLASSES,
        "description": "Domain shift: train on private, test on internet raw"
    },
    3: {
        "name": "exp3_privato_internet_cropped",
        "train_dataset": "privato",
        "test_dataset": "internet_cropped",
        "num_classes": 6,
        "test_classes": INTERNET_CLASSES,
        "description": "Domain shift: train on private, test on internet cropped"
    },
    4: {
        "name": "exp4_combined_internet_cropped",
        "train_dataset": "combined",
        "test_dataset": "internet_cropped",
        "num_classes": 6,
        "test_classes": INTERNET_CLASSES,
        "description": "Best on internet: train on combined, test on internet cropped"
    },
    5: {
        "name": "exp5_combined_privato",
        "train_dataset": "combined",
        "test_dataset": "privato",
        "num_classes": 6,
        "test_classes": ALL_CLASSES,
        "description": "Catastrophic forgetting test: train on combined, test on private"
    }
}

# Default configuration
DEFAULT_CONFIG = {
    "model_variant": "resnet18",
    "epochs": 26,
    "batch_size": 8,
    "learning_rate": 0.001,
    "seed": 57,
    "pretrained": True,
    "num_folds": 5,
}


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
        done_file = exp_dir / f".exp_{exp_id}_done"

        if done_file.exists():
            status = "✓ COMPLETED"
        elif exp_dir.exists():
            status = "⏳ IN PROGRESS"
        else:
            status = "○ NOT STARTED"

        print(f"Exp{exp_id}: {exp_config['name']:40} [{status}]")
        print(f"       {exp_config['description']}")
        print(f"       Train: {exp_config['train_dataset']:15} → Test: {exp_config['test_dataset']}")
        print()

    print("=" * 70 + "\n")


def _generate_experiment_report(results, model_variant, exp, epochs, batch_size,
                                learning_rate, output_path):
    """Genera un report testuale per l'esperimento standard."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"REPORT ESPERIMENTO - {exp['name'].upper()} - {model_variant.upper()}")
    lines.append("=" * 80)
    lines.append(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configurazione
    lines.append("CONFIGURAZIONE")
    lines.append("-" * 50)
    lines.append(f"Esperimento: {exp['name']}")
    lines.append(f"Descrizione: {exp['description']}")
    lines.append(f"Modello: {model_variant}")
    lines.append(f"Train dataset: {exp['train_dataset']}")
    lines.append(f"Test dataset: {exp['test_dataset']}")
    lines.append(f"Epochs: {epochs}")
    lines.append(f"Batch size: {batch_size}")
    lines.append(f"Learning rate: {learning_rate}")
    lines.append(f"Classi test: {exp['test_classes']}")
    lines.append("")

    # Metriche aggregate
    lines.append("=" * 80)
    lines.append("RISULTATI TEST")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Accuracy: {results['accuracy']:.4f}")
    lines.append(f"F1 Macro: {results['f1_macro']:.4f}")
    lines.append(f"Precision Macro: {results.get('precision_macro', 0):.4f}")
    lines.append(f"Recall Macro: {results.get('recall_macro', 0):.4f}")
    lines.append("")

    # Metriche per classe
    if "per_class_metrics" in results:
        lines.append("METRICHE PER CLASSE:")
        lines.append("-" * 70)
        lines.append(f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 70)
        for cls_name, cls_metrics in results["per_class_metrics"].items():
            lines.append(
                f"{cls_name:<25} "
                f"{cls_metrics['precision']:>10.4f} "
                f"{cls_metrics['recall']:>10.4f} "
                f"{cls_metrics['f1']:>10.4f} "
                f"{cls_metrics['support']:>10}"
            )
        lines.append("")

    # Confusion matrix
    if "confusion_matrix" in results and "per_class_metrics" in results:
        class_names = list(results["per_class_metrics"].keys())
        cm = results["confusion_matrix"]
        lines.append("CONFUSION MATRIX (righe = true, colonne = predicted):")
        lines.append("-" * 70)
        header = f"{'':>25}"
        for lbl in class_names:
            header += f" {lbl[:8]:>8}"
        lines.append(header)
        for i, row in enumerate(cm):
            row_str = f"{class_names[i]:>25}"
            for val in row:
                row_str += f" {val:>8}"
            lines.append(row_str)
        lines.append("")

    lines.append("=" * 80)
    lines.append("FINE REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def run_experiment(
    exp_id: int,
    phases: list,
    model_variant: Literal["resnet18", "resnet50"] = "resnet18",
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
        model_variant: Model variant ('resnet18' or 'resnet50')
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        pretrained: Use pretrained weights
        num_folds: Number of folds for cross-validation
    """
    from torch.utils.data import DataLoader
    from app.core.models.resnet import ResNetConfig, ResNetClassifier
    from app.core.training import train, cross_validate
    from app.core.evaluation import evaluate
    from app.preprocessing.dataset_tool import getDatasetFromFile

    exp = EXPERIMENTS[exp_id]
    device = get_device()

    # Per-experiment log file (dentro experiments/)
    results_dir = PROJECT_ROOT / "experiments" / (f"{exp['name']}_{tag}" if tag else exp["name"])
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_log = str(results_dir / f'{exp["name"]}_{model_variant}.log')
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

        # Model config
        config = ResNetConfig(
            variant=model_variant,
            num_classes=exp["num_classes"],
            pretrained=pretrained
        )

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
                    fold_index=fold_index,
                    aggregate_only=aggregate_only,
                    patience=patience,
                )
            else:
                # Cross-validation originale su dataset pre-augmentato
                train_dataset = getDatasetFromFile(train_path)

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
                import json
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

            model = ResNetClassifier(config)

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
                model = ResNetClassifier(config)
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
            import json
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
            try:
                from app.explainability import generate_output_images
                from app.preprocessing.dataset_tool import ImageFolderWithName
                from app.preprocessing.normalization import get_mean, get_std
                from torchvision import transforms as T
                import json

                from app.core.models.resnet import ResNetConfig, ResNetClassifier

                # Determine model path based on experiment type
                if exp_id == 4:
                    # K-fold experiment: use best fold model
                    kfold_exp_dir = str(results_dir) + f"_kfold_{model_variant}"
                    kfold_results_path = os.path.join(kfold_exp_dir, "results.json")
                    if not os.path.exists(kfold_results_path):
                        logging.warning("--explain requires completed k-fold results. Run with --cross-validate first")
                        return
                    with open(kfold_results_path) as f:
                        kfold_data = json.load(f)
                    best_fold = kfold_data.get("best_fold", 0)
                    best_model_path = os.path.join(kfold_exp_dir, f"best_fold_{best_fold}.pt")
                else:
                    # Standard experiment: use last trained model
                    best_model_path = str(model_dir / "model.pt")

                if not os.path.exists(best_model_path):
                    logging.warning(f"Model not found: {best_model_path}")
                    return

                config = ResNetConfig(variant=model_variant, num_classes=exp["num_classes"], pretrained=True)
                model = ResNetClassifier(config)
                model.load_weights(best_model_path)
                model = model.to(device)

                transform = T.Compose([
                    T.Resize((512, 256)),
                    T.ToTensor(),
                    T.Normalize(mean=get_mean(), std=get_std()),
                ])
                test_raw_dir = str(PROJECT_ROOT / DATASET_PATHS[exp["test_dataset"]]["raw"])
                test_dataset = ImageFolderWithName(test_raw_dir, transform=transform)
                explain_dir = str(exp_dir / "explainability")
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
        done_file = exp_dir / f".exp_{exp_id}_done"
        done_file.write_text(f"Completed at {datetime.now().isoformat()}\nModel: {model_variant}\n")

        logging.info(f"\n✓ EXPERIMENT {exp_id} COMPLETED")
        return True

    finally:
        # Remove per-experiment log handler
        logging.getLogger().removeHandler(exp_handler)
        exp_handler.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Orchid Classification Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --status                      # Check status
  python cli.py --train --exp 4               # Train exp4 with ResNet18
  python cli.py --train --exp 4 --model resnet50  # Train exp4 with ResNet50
  python cli.py --cross-validate --exp 4 --folds 5  # 5-fold CV on exp4
  python cli.py --all --model resnet50        # All exps with ResNet50
        """
    )

    # Action flags
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--cross-validate", action="store_true", help="Run k-fold cross validation")
    parser.add_argument("--status", action="store_true", help="Check experiment status")
    parser.add_argument("--explain", action="store_true",
                        help="Generate explanation images (Occlusion, IntegratedGradients) after testing")

    # Model selection
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"],
                        help="Model variant (default: resnet18)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG['epochs'],
                        help=f"Number of epochs (default: {DEFAULT_CONFIG['epochs']})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG['batch_size'],
                        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience on val_loss (0 = disabled, default: 0)")

    # Cross-validation
    parser.add_argument("--folds", type=int, default=DEFAULT_CONFIG['num_folds'],
                        help=f"Number of folds for cross-validation (default: {DEFAULT_CONFIG['num_folds']})")
    parser.add_argument("--fold", type=int, metavar="N",
                        help="Run only fold N (1-based). Saves results for later aggregation.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Aggregate completed folds and run final test (no training).")

    # Experiment selection
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run specific experiment (1-5)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag suffix for output directory (e.g. --tag 100ep creates exp4_..._100ep/)")

    return parser.parse_args()


def main():
    """Main CLI entrypoint."""
    args = parse_args()

    # Status check - no logging setup needed
    if args.status:
        show_status()
        return 0

    # Validate arguments
    if not any([args.all, args.train, args.test, args.cross_validate, args.explain]):
        print("Error: Specify an action (--all, --train, --test, --cross-validate, --explain, --status)")
        return 1

    # Setup logging
    setup_logging()

    # Determine phases to run
    if args.all:
        phases = ['train', 'test']
    else:
        phases = []
        if args.train:
            phases.append('train')
        if args.test:
            phases.append('test')
        if args.cross_validate:
            phases.append('cross-validate')
        if args.explain:
            phases.append('explain')

    # Determine which experiments to run
    if args.exp:
        exp_ids = [args.exp]
    elif args.all:
        exp_ids = list(EXPERIMENTS.keys())
    else:
        print("Error: Specify --exp ID or --all to run all experiments")
        return 1

    # Log configuration
    logging.info("=" * 70)
    logging.info("ORCHID CLASSIFICATION EXPERIMENTS")
    logging.info("=" * 70)
    logging.info(f"Model: {args.model.upper()}")
    logging.info(f"Experiments: {exp_ids}")
    logging.info(f"Phases: {phases}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    if args.patience > 0:
        logging.info(f"Early stopping patience: {args.patience}")
    if "cross-validate" in phases:
        logging.info(f"K-Folds: {args.folds}")
    logging.info("")

    # Run experiments
    success_count = 0
    for exp_id in exp_ids:
        try:
            run_experiment(
                exp_id=exp_id,
                phases=phases,
                model_variant=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                num_folds=args.folds,
                fold_index=args.fold,
                aggregate_only=args.aggregate_only,
                patience=args.patience,
                tag=args.tag,
            )
            success_count += 1
        except Exception as e:
            logging.error(f"Failed to run experiment {exp_id}: {e}")
            import traceback
            traceback.print_exc()
            if len(exp_ids) == 1:
                return 1

    logging.info("\n" + "=" * 70)
    logging.info(f"COMPLETED: {success_count}/{len(exp_ids)} experiments")
    logging.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
