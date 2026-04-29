"""
Upload model results to HuggingFace.

Usage:
    # Upload a specific model from Serie 1
    python scripts/upload_models_to_hf.py --model resnet18
    python scripts/upload_models_to_hf.py --model resnet50

    # Upload all completed models
    python scripts/upload_models_to_hf.py --all

Requires: huggingface_hub, HF token configured (huggingface-cli login)

Repo: marcellorusso/orchid-ncd-models
Structure:
    serie1_ce/resnet18/
    serie1_ce/resnet50/
    ...
    serie2_supcon/resnet18/
    ...
"""
import argparse
import json
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "marcellorusso/orchid-ncd-models"
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments")

SERIE1_MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]
SERIE2_MODELS = []  # TBD


def is_model_complete(tag):
    """Check if a model has all 5 folds + results.json."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, f"exp6_clean_split_{tag}")
    if not os.path.exists(exp_dir):
        return False
    for i in range(5):
        if not os.path.exists(os.path.join(exp_dir, f"fold_{i}.json")):
            return False
    return os.path.exists(os.path.join(exp_dir, "results.json"))


def upload_model(api, tag, serie="serie1_ce"):
    """Upload a single model's experiment results."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, f"exp6_clean_split_{tag}")
    if not os.path.exists(exp_dir):
        print(f"SKIP: {exp_dir} not found")
        return False

    exp_name = os.path.basename(exp_dir)
    remote_dir = f"{serie}/{exp_name}"
    print(f"Uploading {exp_name} -> {remote_dir}/")

    # Upload JSON files (fold results, config, results)
    for fname in os.listdir(exp_dir):
        fpath = os.path.join(exp_dir, fname)
        if fname.endswith(".json"):
            print(f"  {fname}")
            api.upload_file(
                repo_id=REPO_ID,
                path_or_fileobj=fpath,
                path_in_repo=f"{remote_dir}/{fname}",
            )

    # Upload model weights
    for fname in os.listdir(exp_dir):
        fpath = os.path.join(exp_dir, fname)
        if fname.endswith(".pt") and fname.startswith("best_fold"):
            print(f"  {fname}")
            api.upload_file(
                repo_id=REPO_ID,
                path_or_fileobj=fpath,
                path_in_repo=f"{remote_dir}/{fname}",
            )
        elif fname.endswith("_adapter") and os.path.isdir(fpath):
            print(f"  {fname}/")
            api.upload_folder(
                repo_id=REPO_ID,
                folder_path=fpath,
                path_in_repo=f"{remote_dir}/{fname}",
            )

    print(f"  {tag} uploaded")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload model results to HuggingFace")
    parser.add_argument("--model", type=str, help="Model tag to upload (e.g., resnet18)")
    parser.add_argument("--all", action="store_true", help="Upload all completed models")
    parser.add_argument("--serie", type=str, default="serie1_ce",
                        choices=["serie1_ce", "serie2_supcon"],
                        help="Which serie to upload to")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    api = HfApi()

    if not args.dry_run:
        try:
            create_repo(REPO_ID, exist_ok=True)
            print(f"Repo {REPO_ID} ready")
        except Exception as e:
            print(f"Repo note: {e}")

        # Upload README
        readme_path = os.path.join(os.path.dirname(__file__), "hf_models_readme.md")
        if os.path.exists(readme_path):
            api.upload_file(
                repo_id=REPO_ID,
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
            )
            print("README.md uploaded")

    if args.all:
        models = SERIE1_MODELS if args.serie == "serie1_ce" else SERIE2_MODELS
        for tag in models:
            if is_model_complete(tag):
                if args.dry_run:
                    print(f"WOULD UPLOAD: {tag} (complete)")
                else:
                    upload_model(api, tag, args.serie)
            else:
                print(f"SKIP: {tag} (not complete)")
    elif args.model:
        if args.dry_run:
            print(f"WOULD UPLOAD: {args.model}")
            exp_dir = os.path.join(EXPERIMENTS_DIR, f"exp6_clean_split_{args.model}")
            if os.path.exists(exp_dir):
                for f in sorted(os.listdir(exp_dir)):
                    if f.endswith(".json") or (f.endswith(".pt") and f.startswith("best_fold")):
                        print(f"  {f}")
        else:
            upload_model(api, args.model, args.serie)
    else:
        # Show status
        print("Model status:")
        for tag in SERIE1_MODELS:
            exp_dir = os.path.join(EXPERIMENTS_DIR, f"exp6_clean_split_{tag}")
            if os.path.exists(exp_dir):
                folds = sum(1 for i in range(5) if os.path.exists(os.path.join(exp_dir, f"fold_{i}.json")))
                has_results = os.path.exists(os.path.join(exp_dir, "results.json"))
                status = "COMPLETE" if folds == 5 and has_results else f"{folds}/5 folds"
                print(f"  {tag}: {status}")
            else:
                print(f"  {tag}: not started")

    print("Done")


if __name__ == "__main__":
    main()
