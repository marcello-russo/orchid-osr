"""Experiment + dataset sync — importable logic (CLI: `orchid hf sync|download`)."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from app.config import settings
from app.core.hf_sync import download_file, list_repo_files

logger = logging.getLogger(__name__)


def _is_essential(path: str) -> bool:
    """Filter files needed for the explorer dashboard and latent space viewer."""
    parts = path.split("/")
    if parts[0] in ("old",):
        return False
    for skip in ("ncd_figures", ".cache"):
        if skip in parts:
            return False
    if path.endswith(".json"):
        return True
    filename = parts[-1]
    # best-fold serving weights (classify) + LoRA adapters
    if filename.startswith("best_fold_") and filename.endswith(".pt"):
        return True
    if any("_adapter" in p for p in parts):
        return True
    if "embeddings" in parts:
        if filename.endswith((".npy", ".json")):
            return True
    return False


def sync_experiments(
    dst: str | None = None,
    repo: str | None = None,
    api: HfApi | None = None,
) -> dict[str, int]:
    """Download essential experiment JSONs + embeddings. Returns counters."""
    token = os.environ.get("HF_TOKEN") or settings.hf_token or None
    if not token:
        print("No HF_TOKEN, skipping experiment sync")
        return {"synced": 0, "skipped": 0, "errors": 0}
    client = api or HfApi(token=token)
    repo_id = repo or settings.hf_models_repo
    dst_dir = dst or os.environ.get("EXPERIMENTS_DIR", "/data/experiments")
    os.makedirs(dst_dir, exist_ok=True)

    all_paths = list_repo_files(client, repo_id)
    if not all_paths:
        print("Failed to list repo files (empty or error)")
        return {"synced": 0, "skipped": 0, "errors": 0}

    essential = [p for p in all_paths if _is_essential(p)]
    print(f"Found {len(all_paths)} files total, syncing {len(essential)} essential files (JSONs + embeddings)...")

    count, skipped, errors = 0, 0, 0
    for src in essential:
        parts = src.split("/")
        flat = src if len(parts) < 2 else "/".join(parts[1:])
        target = os.path.join(dst_dir, flat)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        src_mtime = None
        try:
            repo_info = client.get_paths_info(repo_id, src)
            if repo_info:
                src_mtime = repo_info[0].last_commit_date.timestamp() if repo_info[0].last_commit_date else None
        except Exception:
            pass
        dst_mtime = os.path.getmtime(target) if os.path.exists(target) else None
        if src_mtime and dst_mtime and dst_mtime >= src_mtime:
            skipped += 1
            continue

        try:
            download_file(client, repo_id, src, target, token)
            count += 1
        except Exception as e:
            errors += 1
            print(f"  WARN: failed {src}: {e}")

    print(f"Synced {count} files (skipped {skipped} up-to-date, {errors} errors)")

    index_path = os.path.join(dst_dir, "_index.json")
    models_found = []
    for rj in Path(dst_dir).rglob("results.json"):
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
    return {"synced": count, "skipped": skipped, "errors": errors}


def download_dataset(
    sample_ratio: float = 1.0,
    dataset_dir: str = "dataset",
    repo: str | None = None,
) -> bool:
    """Download + reshape the cropped dataset. Returns True on success."""
    from app.config import settings as _s

    token = os.environ.get("HF_TOKEN") or _s.hf_token or None
    if not token:
        print("Warning: HF_TOKEN not found. Downloads might hit rate limits or fail if repos are private.")
    repo_id = repo or _s.hf_dataset_repo
    print("Downloading dataset from HuggingFace...")
    try:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            local_dir=dataset_dir,
            allow_patterns=[
                "cropped/*/train/**",
                "cropped/*/test/**",
                "cropped/*/test_clean/**",
            ],
            token=token,
        )
        for size in ["256x512", "224x224"]:
            for hf_split in ["train", "test", "test_clean"]:
                src = os.path.join(dataset_dir, "cropped", size, hf_split)
                dst = os.path.join(dataset_dir, f"cropped_{size}", hf_split.replace("_clean", ""))
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    for item in os.listdir(src):
                        s = os.path.join(src, item)
                        t = os.path.join(dst, item)
                        if os.path.isdir(s):
                            shutil.copytree(s, t, dirs_exist_ok=True)
                    shutil.rmtree(src)

        if sample_ratio < 1.0:
            import random

            print(f"Dataset sampling attivo: ratio={sample_ratio}")
            for size in ["256x512", "224x224"]:
                for split in ["train", "test"]:
                    p = os.path.join(dataset_dir, f"cropped_{size}", split)
                    if not os.path.isdir(p):
                        continue
                    for class_dir in sorted(os.listdir(p)):
                        cd = os.path.join(p, class_dir)
                        if not os.path.isdir(cd):
                            continue
                        files = [f for f in os.listdir(cd) if os.path.isfile(os.path.join(cd, f))]
                        random.shuffle(files)
                        for f in files[max(1, int(len(files) * sample_ratio)) :]:
                            os.unlink(os.path.join(cd, f))
        print("Dataset downloaded successfully.")
        return True
    except Exception as e:
        print(f"Dataset download failed: {e}")
        return False


def main() -> None:
    sync_experiments()


if __name__ == "__main__":
    main()
