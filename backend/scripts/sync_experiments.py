"""Download experiment metadata from HuggingFace models repo at startup.

Production-grade sync:
- Downloads all JSON files (needed for explorer dashboard)
- Does NOT download .pt weights (handled on-demand by weights.py)
- Skips embeddings/, ncd_figures/, .cache/ (not needed for inference)
- Uses atomic writes to prevent corruption on partial downloads
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from app.core.hf_sync import download_file, list_repo_files


def _is_essential(path: str) -> bool:
    """Filter: only JSON files (metadata), skip heavy artifacts."""
    if not path.endswith(".json"):
        return False
    parts = path.split("/")
    # Skip embeddings, ncd_figures, .cache directories
    for skip in ("embeddings", "ncd_figures", ".cache"):
        if skip in parts:
            return False
    return True


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("No HF_TOKEN, skipping experiment sync")
        return

    api = HfApi(token=token)
    REPO = "marcellorusso/orchid-ncd-models"
    DST = os.environ.get("EXPERIMENTS_DIR", "/data/experiments")
    os.makedirs(DST, exist_ok=True)

    all_paths = list_repo_files(api, REPO)
    if not all_paths:
        print("Failed to list repo files (empty or error)")
        return

    essential = [p for p in all_paths if _is_essential(p)]
    n_json = len(essential)
    print(f"Found {len(all_paths)} files total, syncing {n_json} essential JSONs...")

    count = 0
    skipped = 0
    errors = 0

    for src in essential:
        parts = src.split("/")
        if parts[0] in ("old",):
            skipped += 1
            continue

        if len(parts) < 2:
            flat = src
        else:
            flat = "/".join(parts[1:])
        dst = os.path.join(DST, flat)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        src_mtime = None
        dst_mtime = None
        try:
            # Check remote file metadata
            repo_info = api.get_paths_info(REPO, src)
            if repo_info:
                src_mtime = repo_info[0].last_commit_date.timestamp() if repo_info[0].last_commit_date else None
        except Exception:
            pass

        if os.path.exists(dst):
            dst_mtime = os.path.getmtime(dst)

        # Skip if local is newer or same age as remote
        if src_mtime and dst_mtime and dst_mtime >= src_mtime:
            skipped += 1
            continue

        try:
            download_file(api, REPO, src, dst, token)
            count += 1
        except Exception as e:
            errors += 1
            print(f"  WARN: failed {src}: {e}")

    print(f"Synced {count} files (skipped {skipped} up-to-date, {errors} errors)")

    # Generate a lightweight index for fast model discovery
    index_path = os.path.join(DST, "_index.json")
    models_found = []
    for rj in Path(DST).rglob("results.json"):
        try:
            with open(rj) as f:
                data = json.load(f)
            models_found.append(
                {
                    "dir": rj.parent.name,
                    "model": data.get("model_variant"),
                    "exp": data.get("exp_num"),
                    "f1": data.get("mean_f1") or data.get("test_metrics", {}).get("f1_macro"),
                    "best_fold": data.get("best_fold"),
                }
            )
        except Exception:
            continue
    with open(index_path, "w") as f:
        json.dump({"models": models_found, "count": len(models_found)}, f)
    print(f"Index written: {len(models_found)} models")


if __name__ == "__main__":
    main()
