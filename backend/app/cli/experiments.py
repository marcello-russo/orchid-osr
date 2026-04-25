# app/cli/experiments.py
"""
Thin CLI entrypoint: parse_args() + main() only.

All experiment logic lives in:
  - definitions.py  — constants and experiment definitions
  - reports.py      — report generation
  - runner.py       — run_experiment(), setup_logging(), get_device(), show_status()
"""
import argparse
import logging
import os
import sys

from app.core.models import available_models
from app.cli.definitions import EXPERIMENTS, DEFAULT_CONFIG
from app.cli.runner import run_experiment, setup_logging, show_status



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
    parser.add_argument("--plot", type=str, metavar="PATH",
                        help="Generate plots for experiment directory (path to exp dir, or 'all')")

    # Model selection
    parser.add_argument("--model", type=str, default="resnet18",
                        help=f"Model variant (default: resnet18). Available: {', '.join(available_models())}")

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

    # Validate model choice
    valid_models = available_models()
    if args.model not in valid_models:
        print(f"Error: Unknown model '{args.model}'. Available: {', '.join(valid_models)}")
        return 1

    # Status check - no logging setup needed
    if args.status:
        show_status()
        return 0

    # Plot mode
    if args.plot:
        from app.visualization import plot_experiment
        if args.plot.lower() == "all":
            from glob import glob
            exp_base = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
            exp_base = os.path.abspath(exp_base)
            experiment_dirs = {
                os.path.dirname(fold_file)
                for fold_file in glob(os.path.join(exp_base, "*", "fold_*.json"))
            }
            for exp_d in sorted(experiment_dirs):
                print(f"\n{'=' * 50}")
                plot_experiment(exp_d)
        else:
            plot_experiment(args.plot)
        return 0

    # Validate arguments
    if not any([args.all, args.train, args.test, args.cross_validate, args.explain, args.plot]):
        print("Error: Specify an action (--all, --train, --test, --cross-validate, --explain, --plot, --status)")
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
