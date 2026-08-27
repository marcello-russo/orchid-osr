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
from pathlib import Path
import sys
from app.cli.definitions import DEFAULT_CONFIG, EXPERIMENTS, PROJECT_ROOT, TRAIN_CLASSES
from app.cli.runner import run_experiment, setup_logging, show_status
from app.core.models import available_models

logger = logging.getLogger(__name__)


def _active_classes_for_exp(exp_id: int) -> list[str] | None:
    """Classi attive per un esperimento (TRAIN_CLASSES meno exclude_classes).

    Per Exp 4, garganica è esclusa: gli indici 0-4 sono riassegnati alle
    classi residue. Nessun'altra exp ha exclude_classes → None (6 classi).
    """
    cfg = EXPERIMENTS.get(exp_id, {})
    exclude = cfg.get("exclude_classes") or []
    if not exclude:
        return None
    return [c for c in TRAIN_CLASSES if c not in set(exclude)]


def _build_pair_weights(hard_weight: float, active_classes: list[str] | None = None) -> dict:
    """Build pair_weights dict from class names (not hardcoded indices).

    active_classes: lista delle classi effettivamente usate nel training
    (dopo eventuale exclude_classes). Se None, usa TRAIN_CLASSES (6).
    Necessario per Exp 4, dove garganica è esclusa e gli indici 0-4
    sono riassegnati alle 5 classi residue.
    """
    cls_list = active_classes if active_classes is not None else TRAIN_CLASSES
    cls_map = {name: i for i, name in enumerate(cls_list)}
    result = {}

    def _add(a, b, weight):
        i, j = cls_map.get(a), cls_map.get(b)
        if i is not None and j is not None:
            result[(i, j)] = weight
            result[(j, i)] = weight

    _add("O. majellensis", "O. sphegodes", hard_weight)
    _add("O. sphegodes", "O. sphegodes_Palena", hard_weight * 0.6)
    return result


def _handle_auth_commands(args) -> int | None:
    """Handle user management commands. Returns exit code or None if not an auth command."""
    if not any([args.list_users, args.gen_invites is not None, args.add_user is not None]):
        return None
    from app.core.auth import (
        generate_invite,
        list_users_and_invites,
        hash_password,
        _load_data,
        _save_data,
        ensure_admin_exists,
    )

    ensure_admin_exists()
    if args.list_users:
        data = list_users_and_invites()
        print("=== Users ===")
        for u in data["users"]:
            print(f"  {u['username']} ({u['role']})")
        print("\n=== Invites ===")
        for i in data["invites"]:
            status = "used" if i["used"] else "available"
            print(f"  {i['code']} [{status}] ({i['role']})")
        return 0
    if args.gen_invites is not None:
        codes = generate_invite(count=args.gen_invites)
        for c in codes:
            print(c)
        return 0
    if args.add_user is not None:
        if not args.password:
            print("--password is required with --add-user")
            return 1
        username = args.add_user or args.username
        if not username:
            print("Specify username with --add-user <name> or --username")
            return 1
        data = _load_data()
        if any(u["username"] == username for u in data["users"]):
            print(f"User '{username}' already exists")
            return 1
        data["users"].append(
            {
                "username": username,
                "password_hash": hash_password(args.password),
                "role": args.role,
            }
        )
        _save_data(data)
        print(f"User '{username}' created with role '{args.role}'")
        return 0
    return None


def _run_crop_step(args, cropped_dir: str, internet_dir: str, cropper_model: str) -> None:
    """Run cropping step of preprocessing pipeline."""
    from app.cli.definitions import PROJECT_ROOT
    from app.config import settings as cfg
    from app.preprocessing.cropper import crop_dataset, crop_single_image, load_cropper_model

    logger.info("=" * 60)
    logger.info("PIPELINE STEP 1: CROP")
    logger.info("=" * 60)

    raw_private = str(PROJECT_ROOT / "dataset" / "raw")
    if os.path.exists(raw_private):
        logger.info("Cropping private raw images...")
        crop_dataset(raw_private, cropped_dir, cropper_model, score_threshold=args.score_threshold)

    if os.path.exists(internet_dir):
        logger.info("Cropping internet raw images...")
        model = load_cropper_model(cropper_model)
        for inet_class_dir in sorted(Path(internet_dir).iterdir()):
            if not inet_class_dir.is_dir():
                continue
            inet_name = inet_class_dir.name
            class_name = None
            for cls in cfg.class_names:
                if cls.replace(" ", "") == inet_name:
                    class_name = cls
                    break
            if class_name is None:
                logger.warning(f"Skipping {inet_name}: no matching class in config")
                continue
            out_dir = os.path.join(cropped_dir, class_name)
            os.makedirs(out_dir, exist_ok=True)
            files = sorted(f for f in os.listdir(inet_class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
            cropped_count = 0
            for img_file in files:
                img_path = os.path.join(str(inet_class_dir), img_file)
                cropped_img, _ = crop_single_image(model, img_path, args.score_threshold, fallback="full")
                if cropped_img is not None:
                    cropped_img.save(os.path.join(out_dir, img_file), quality=95)
                    cropped_count += 1
            logger.info(f"Internet {inet_name} → {class_name}: {cropped_count}/{len(files)}")


def _run_split_step(cropped_dir: str, internet_dir: str) -> None:
    """Run train/test split step."""
    from app.cli.definitions import PROJECT_ROOT
    from app.config import settings as cfg
    from app.preprocessing.split import create_clean_split

    logger.info("=" * 60)
    logger.info("PIPELINE STEP 2: SPLIT")
    logger.info("=" * 60)
    create_clean_split(
        cropped_dir=cropped_dir,
        internet_dir=internet_dir,
        output_dir=os.path.join(str(PROJECT_ROOT), "dataset", "cropped"),
        class_names=cfg.class_names,
        internet_classes=cfg.internet_classes,
        test_per_class=cfg.test_per_class,
        seed=cfg.split_seed,
    )


def _run_resize_step() -> None:
    """Run resize step for all architectures."""
    from app.cli.definitions import PROJECT_ROOT
    from app.preprocessing.resize_dataset import resize_for_all_architectures

    logger.info("=" * 60)
    logger.info("PIPELINE STEP 3: RESIZE")
    logger.info("=" * 60)
    cropped_base = str(PROJECT_ROOT / "dataset" / "cropped")
    dataset_base = str(PROJECT_ROOT / "dataset")
    for split_name in ["train", "test"]:
        split_dir = os.path.join(cropped_base, split_name)
        if not os.path.exists(split_dir):
            logger.warning(f"Skipping {split_name} — directory not found")
            continue
        logger.info(f"Resizing {split_name}...")
        resize_for_all_architectures(split_dir, dataset_base, split_name=split_name)


def _run_pipeline(args) -> None:
    """Run preprocessing pipeline (crop/split/resize)."""
    from app.cli.definitions import PROJECT_ROOT

    setup_logging()
    cropped_dir = str(PROJECT_ROOT / "dataset" / "cropped")
    internet_dir = os.path.join(str(PROJECT_ROOT), args.internet_dir)
    cropper_model = os.path.join(str(PROJECT_ROOT), args.cropper_model)

    if args.crop or args.pipeline:
        _run_crop_step(args, cropped_dir, internet_dir, cropper_model)

    if args.split or args.pipeline:
        _run_split_step(cropped_dir, internet_dir)

    if args.resize or args.pipeline:
        _run_resize_step()


def _run_plot_mode(args) -> None:
    """Generate plots for experiment directories."""
    from app.visualization import plot_experiment
    from glob import glob

    if args.plot.lower() == "all":
        exp_base = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
        exp_base = os.path.abspath(exp_base)
        experiment_dirs = {os.path.dirname(f) for f in glob(os.path.join(exp_base, "*", "fold_*.json"))}
        for exp_d in sorted(experiment_dirs):
            print(f"\n{'=' * 50}")
            plot_experiment(exp_d)
    else:
        plot_experiment(args.plot)


def _resolve_phases(args) -> list[str]:
    """Determine which phases to run based on args."""
    if args.embeddings:
        if any([args.train, args.test, args.cross_validate, args.explain]):
            logging.warning("--embeddings is standalone; other phase flags will be ignored")
        return ["embeddings"]
    if args.embeddings_missing:
        if any(
            [
                args.train,
                args.test,
                args.cross_validate,
                args.explain,
                args.embeddings,
                args.ncd,
                args.ncd_comparison,
            ]
        ):
            logging.warning("--embeddings-missing is standalone; other phase flags will be ignored")
        return ["embeddings_missing"]
    if args.ncd:
        if any([args.train, args.test, args.cross_validate, args.explain, args.embeddings, args.ncd_comparison]):
            logging.warning("--ncd is standalone; other phase flags will be ignored")
        return ["ncd"]
    if args.ncd_comparison:
        if any([args.train, args.test, args.cross_validate, args.explain, args.embeddings, args.ncd]):
            logging.warning("--ncd-comparison is standalone; other phase flags will be ignored")
        return ["ncd_comparison"]
    if args.all:
        return ["train", "test"]
    phases = []
    if args.train:
        phases.append("train")
    if args.test:
        phases.append("test")
    if args.cross_validate:
        phases.append("cross-validate")
    if args.explain:
        phases.append("explain")
    return phases


def _resolve_experiments(args, phases: list[str]) -> list[int] | None:
    """Determine which experiment IDs to run."""
    if args.ncd_comparison:
        return []
    if args.exp:
        return [args.exp]
    if args.all:
        from app.cli.definitions import EXPERIMENTS

        return list(EXPERIMENTS.keys())
    print("Error: Specify --exp ID or --all to run all experiments")
    return None


def _log_config(args, phases: list[str], exp_ids: list[int]) -> None:
    """Log experiment configuration."""
    is_inference = set(phases).issubset({"ncd", "embeddings"})

    logging.info("=" * 70)
    logging.info("ORCHID CLASSIFICATION EXPERIMENTS")
    logging.info("=" * 70)
    logging.info(f"Model: {args.model.upper()}")
    logging.info(f"Experiments: {exp_ids}")
    logging.info(f"Phases: {phases}")
    if is_inference:
        logging.info("Epochs: N/A (inference-only phase)")
        logging.info("Batch size: N/A")
        logging.info("Learning rate: N/A")
    else:
        logging.info(f"Epochs: {args.epochs}")
        logging.info(f"Batch size: {args.batch_size}")
        logging.info(f"Learning rate: {args.lr}")
    if args.patience > 0:
        logging.info(f"Early stopping patience: {args.patience}")
    if "cross-validate" in phases:
        logging.info(f"K-Folds: {args.folds}")
    logging.info("")


def _run_embeddings_missing(args, valid_models: list[str]) -> int:
    """Scan experiments and generate missing embeddings."""
    from app.cli.definitions import ALL_CLASSES, PROJECT_ROOT
    from app.core.models import get_registry_entry
    from app.core.ncd.extract import run_embedding_extraction
    from app.core.hf_sync import parse_exp_dir_name
    import json
    import torch

    if not torch.cuda.is_available():
        logging.error("--embeddings-missing requires CUDA (GPU). Aborting.")
        return 1

    exp_base = str(PROJECT_ROOT / "experiments")
    model_set = set(valid_models)
    skipped = 0
    generated = 0

    for d in sorted(os.listdir(exp_base)):
        full_path = os.path.join(exp_base, d)
        if not os.path.isdir(full_path) or not d.startswith("exp"):
            continue

        exp_id, model_variant = parse_exp_dir_name(d, model_set)
        if exp_id is None or model_variant is None:
            logging.warning(f"Skipping {d}: could not parse exp/model")
            skipped += 1
            continue

        if exp_id not in EXPERIMENTS:
            logging.info(f"Skipping {d}: exp_id {exp_id} not in EXPERIMENTS")
            skipped += 1
            continue

        results_path = os.path.join(full_path, "results.json")
        if not os.path.exists(results_path):
            logging.info(f"SKIP {d}: no results.json (run sync-local first)")
            skipped += 1
            continue

        with open(results_path) as f:
            results = json.load(f)
        best_fold = results.get("best_fold")
        if best_fold is None:
            logging.info(f"SKIP {d}: no best_fold in results.json")
            skipped += 1
            continue

        best_pt = os.path.join(full_path, f"best_fold_{best_fold}.pt")
        if not os.path.exists(best_pt):
            logging.info(f"SKIP {d}: best_fold_{best_fold}.pt not found")
            skipped += 1
            continue

        emb_dir = os.path.join(full_path, "embeddings")
        if os.path.exists(os.path.join(emb_dir, "embeddings.npy")):
            logging.info(f"OK   {d}: embeddings already present")
            continue

        logging.info(f"GEN  {d}: generating embeddings...")
        try:
            config = EXPERIMENTS[exp_id]
            entry = get_registry_entry(model_variant)
            w, h = entry.input_size
            data_dir = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / "train")
            if not os.path.exists(data_dir):
                logging.warning(f"Train dataset not found: {data_dir}, trying test dir...")
                data_dir = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / "test")
                if not os.path.exists(data_dir):
                    logging.error(f"Dataset not found for {d}, skipping")
                    skipped += 1
                    continue
            run_embedding_extraction(
                exp_dir=full_path,
                model_variant=model_variant,
                num_classes=config["num_classes"],
                class_names=config.get("test_classes", ALL_CLASSES),
                data_dir=data_dir,
                batch_size=8,
            )
            generated += 1
        except Exception as e:
            logging.error(f"FAIL {d}: {e}")
            import traceback

            traceback.print_exc()
            skipped += 1

    print(f"\nEmbeddings generated: {generated}, skipped: {skipped}")
    return 0


def _run_experiment_loop(args, phases: list[str], exp_ids: list[int]) -> int:
    """Run the experiment loop for training/testing/cross-validation."""
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
                gradient_accumulation_steps=args.accumulation,
                proj_dim=args.proj_dim,
                temperature=args.temperature,
                ce_epochs=args.ce_epochs,
                use_prototypes=args.sphor,
                repulse_weight=args.repulse_weight,
                pair_weights=(
                    _build_pair_weights(
                        args.hard_weight,
                        active_classes=_active_classes_for_exp(exp_id),
                    )
                    if args.hard_weight > 0
                    else None
                ),
                full_finetune=args.full_finetune,
                osr_ensemble=args.osr_ensemble,
            )
            success_count += 1
        except Exception as e:
            logging.error(f"Failed to run experiment {exp_id}: {e}")
            import traceback

            traceback.print_exc()
            if len(exp_ids) == 1:
                return 1
    logging.info(f"\nCOMPLETED: {success_count}/{len(exp_ids)} experiments")
    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Orchid Classification Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --status                                    # Check status
  python cli.py --train --exp 4                             # Train exp4 with ResNet18
  python cli.py --train --exp 4 --model resnet50            # Train exp4 with ResNet50
  python cli.py --train --exp 4 --model convnext_tiny       # Train exp4 with ConvNeXt-Tiny
  python cli.py --train --exp 4 --model dinov2_small        # Train exp4 with DINOv2-Small + LoRA
  python cli.py --cross-validate --exp 4 --folds 5          # 5-fold CV on exp4
  python cli.py --all --model convnext_small                # All exps with ConvNeXt-Small
        """,
    )

    # Action flags
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--cross-validate", action="store_true", help="Run k-fold cross validation")
    parser.add_argument("--status", action="store_true", help="Check experiment status")
    parser.add_argument("--crop", action="store_true", help="Crop raw images using Faster R-CNN flower detector")
    parser.add_argument("--split", action="store_true", help="Create clean train/test split from cropped images")
    parser.add_argument(
        "--resize", action="store_true", help="Resize cropped images for all architectures (256x512, 224x224)"
    )
    parser.add_argument(
        "--pipeline", action="store_true", help="Run full preprocessing pipeline: crop → split → resize"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate explanation images (Occlusion, IntegratedGradients) after testing",
    )
    parser.add_argument(
        "--plot", type=str, metavar="PATH", help="Generate plots for experiment directory (path to exp dir, or 'all')"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help=f"Model variant (default: resnet18). Available: {', '.join(available_models())}",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help=f"Number of epochs (default: {DEFAULT_CONFIG['epochs']})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})",
    )
    parser.add_argument(
        "--patience", type=int, default=0, help="Early stopping patience on val_loss (0 = disabled, default: 0)"
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size × accumulation, default: 1)",
    )

    # Cross-validation
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_CONFIG["num_folds"],
        help=f"Number of folds for cross-validation (default: {DEFAULT_CONFIG['num_folds']})",
    )
    parser.add_argument(
        "--fold", type=int, metavar="N", help="Run only fold N (0-based). Saves results for later aggregation."
    )
    parser.add_argument(
        "--aggregate-only", action="store_true", help="Aggregate completed folds and run final test (no training)."
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Extract embeddings + t-SNE/UMAP from trained model and save for visualization.",
    )
    parser.add_argument(
        "--embeddings-missing",
        action="store_true",
        help="Detect experiments with .pt weights but missing embeddings and generate them.",
    )
    parser.add_argument(
        "--ncd", action="store_true", help="Run Novel Class Discovery pipeline: cluster embeddings + evaluate."
    )
    parser.add_argument(
        "--ncd-comparison", action="store_true", help="Generate ncd_comparison.json: cross-model NCD comparison table."
    )

    # Auth management
    parser.add_argument(
        "--add-user", nargs="?", const=None, help="Create user directly (use --username --password --role)"
    )
    parser.add_argument("--username", help="Username for --add-user")
    parser.add_argument("--password", help="Password for --add-user")
    parser.add_argument("--role", default="user", choices=["user", "admin"], help="Role for --add-user")
    parser.add_argument("--gen-invites", nargs="?", type=int, const=1, help="Generate N invite tokens")
    parser.add_argument("--list-users", action="store_true", help="List users and invites")

    # Pipeline parameters
    parser.add_argument(
        "--cropper-model",
        type=str,
        default="models/cropper/fasterrcnn_orchid3.pth",
        help="Path to Faster R-CNN cropper weights",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="Minimum detection confidence for cropper (default: 0.5)"
    )
    parser.add_argument(
        "--internet-dir",
        type=str,
        default=str(PROJECT_ROOT / ".." / ".." / "drive" / "Immagini_Internet"),
        help="Directory with raw internet images (O.{Class} subdirs)",
    )

    # Experiment selection
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3, 4, 5, 10], help="Run specific experiment (1-10)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag suffix for output directory (e.g. --tag 100ep creates exp4_..._100ep/)",
    )
    parser.add_argument(
        "--proj-dim", type=int, default=128, help="SupCon projection head dimension (exp 7+, default: 128)"
    )
    parser.add_argument("--temperature", type=float, default=0.07, help="SupCon temperature (exp 7+, default: 0.07)")
    parser.add_argument(
        "--ce-epochs", type=int, default=50, help="CE fine-tune epochs after SupCon (exp 7+, default: 50)"
    )
    parser.add_argument(
        "--sphor",
        action="store_true",
        help="Enable Spherical Orthogonal prototypes (SpHOR-style) for open-set SupCon training",
    )
    parser.add_argument(
        "--repulse-weight", type=float, default=0.01, help="Weight for prototype repulsion loss (default: 0.01)"
    )
    parser.add_argument(
        "--full-finetune", action="store_true", help="Deep fine-tuning (LoRA++ + partial unfreeze) in Phase 2 (Exp 10)"
    )
    parser.add_argument(
        "--hard-weight",
        type=float,
        default=5.0,
        help="Hard negative mining weight for (majellensis, sphegodes) pair (default: 5.0)",
    )
    parser.add_argument(
        "--osr-ensemble",
        action="store_true",
        help="Use ensemble OSR (Mahalanobis+KNN+Energy) instead of OpenMax (Exp 10)",
    )

    return parser.parse_args()


def main():
    """Main CLI entrypoint."""
    args = parse_args()

    valid_models = available_models()
    if args.model not in valid_models:
        print(f"Error: Unknown model '{args.model}'. Available: {', '.join(valid_models)}")
        return 1

    result = _handle_auth_commands(args)
    if result is not None:
        return result

    if args.status:
        show_status()
        return 0

    if any([args.crop, args.split, args.resize, args.pipeline]):
        _run_pipeline(args)
        if not any([args.all, args.train, args.test, args.cross_validate, args.explain]):
            return 0

    if args.plot:
        _run_plot_mode(args)
        return 0

    setup_logging()
    phases = _resolve_phases(args)
    exp_ids = _resolve_experiments(args, phases)
    if exp_ids is None:
        return 1
    _log_config(args, phases, exp_ids)

    if "ncd_comparison" in phases:
        from scripts.ncd_comparison import main as _ncd_comp

        _ncd_comp()
        return 0

    if "embeddings_missing" in phases:
        return _run_embeddings_missing(args, valid_models)

    return _run_experiment_loop(args, phases, exp_ids)


if __name__ == "__main__":
    sys.exit(main())
