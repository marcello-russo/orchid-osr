"""Upload the OrchID OSR-NCD dataset to HuggingFace.

Structure:
  raw/private/{class}/     — raw private field photos
  raw/internet/{class}/    — raw internet-sourced images
  cropped/256x512/train/   — letterbox-resized train (ResNet)
  cropped/256x512/test/    — letterbox-resized test (ResNet)
  cropped/224x224/train/   — letterbox-resized train (ConvNeXt, DINOv2)
  cropped/224x224/test/    — letterbox-resized test (ConvNeXt, DINOv2)

Usage:
    python scripts/hf_tools/upload_dataset_to_hf.py

Requires: huggingface_hub, HF token configured (huggingface-cli login)
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from huggingface_hub import HfApi, create_repo

from scripts.hf_config import HF_DATASET_REPO

REPO_ID = HF_DATASET_REPO
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
INTERNET_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "..", "drive", "Immagini_Internet"))

SIZES = ["256x512", "224x224"]


def _copy_tree(src, dst):
    if not os.path.isdir(src):
        print(f"  SKIP {src} (not found)")
        return 0
    count = 0
    for cls in sorted(os.listdir(src)):
        src_cls = os.path.join(src, cls)
        if not os.path.isdir(src_cls):
            continue
        dst_cls = os.path.join(dst, cls)
        shutil.copytree(src_cls, dst_cls)
        n = sum(len(f) for _, _, f in os.walk(dst_cls))
        count += n
        print(f"    {cls}: {n} files")
    return count


def main():
    api = HfApi()

    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"Repo {REPO_ID} ready")
    except Exception as e:
        print(f"Repo creation note: {e}")

    staging = tempfile.mkdtemp(prefix="hf_upload_")
    try:
        # --- raw/private ---
        print("\nCopying raw/private...")
        dst = os.path.join(staging, "raw", "private")
        os.makedirs(dst, exist_ok=True)
        _copy_tree(os.path.join(DATASET_DIR, "raw"), dst)

        # --- raw/internet ---
        print("\nCopying raw/internet...")
        dst = os.path.join(staging, "raw", "internet")
        os.makedirs(dst, exist_ok=True)
        if os.path.isdir(INTERNET_DIR):
            for inet_dir in sorted(os.listdir(INTERNET_DIR)):
                inet_path = os.path.join(INTERNET_DIR, inet_dir)
                if not os.path.isdir(inet_path):
                    continue
                if inet_dir == "CROPPED_by_ARUTA":
                    continue
                # e.g. "O.exaltata" → keep as-is on HF
                dst_cls = os.path.join(dst, inet_dir)
                shutil.copytree(inet_path, dst_cls)
                n = sum(len(f) for _, _, f in os.walk(dst_cls))
                print(f"    {inet_dir}: {n} files")

        # --- cropped/{size}/train + test ---
        for size in SIZES:
            cropped_base = os.path.join(DATASET_DIR, f"cropped_{size}")
            for split_src, split_dst in [("train", "train"), ("test", "test")]:
                src = os.path.join(cropped_base, split_src)
                dst = os.path.join(staging, "cropped", size, split_dst)
                if not os.path.isdir(src):
                    print(f"\n  SKIP cropped/{size}/{split_dst} ({src} not found)")
                    continue
                print(f"\nCopying cropped/{size}/{split_dst}...")
                os.makedirs(dst, exist_ok=True)
                _copy_tree(src, dst)

        # --- metadata ---
        manifest = os.path.join(DATASET_DIR, "split_manifest.json")
        if os.path.exists(manifest):
            shutil.copy2(manifest, os.path.join(staging, "split_manifest.json"))

        readme_src = os.path.join(os.path.dirname(__file__), "hf_dataset_readme.md")
        if os.path.exists(readme_src):
            shutil.copy2(readme_src, os.path.join(staging, "README.md"))

        total = sum(len(f) for _, _, f in os.walk(staging))
        print(f"\nUploading {total} files from staging...")

        api.upload_large_folder(
            repo_id=REPO_ID,
            folder_path=staging,
            repo_type="dataset",
        )
        print("Upload complete")

    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\nDone! Dataset at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
