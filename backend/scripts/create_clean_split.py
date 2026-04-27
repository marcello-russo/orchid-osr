"""
Create a clean train/test split with no data leakage.

Rules:
- Internet classes (exaltata, garganica, incubacea, sphegodes):
  50 images from internet_cropped → test, rest → train
  All private images → train
- Private-only classes (majellensis, sphegodes_Palena):
  50 images from private → test, rest → train

Result:
  dataset/train_clean/{class}/ — training images (no overlap with test)
  dataset/test_clean/{class}/  — test images (50 per class, balanced)

Uses fixed seed for reproducibility. Non-destructive (copies, not moves).
"""
import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np

SEED = 42
TEST_PER_CLASS = 50

BASE_DIR = Path(__file__).parent.parent / "dataset"

# Source directories
PRIVATE_DIR = BASE_DIR / "raw"
INTERNET_CROPPED_DIR = BASE_DIR / "internet_cropped"

# Output directories
TRAIN_CLEAN_DIR = BASE_DIR / "train_clean"
TEST_CLEAN_DIR = BASE_DIR / "test_clean"

# All 6 classes
ALL_CLASSES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. majellensis",
    "O. sphegodes",
    "O. sphegodes_Palena",
]

# Classes that have internet images
INTERNET_CLASSES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. sphegodes",
]

# Classes that only have private images
PRIVATE_ONLY_CLASSES = [
    "O. majellensis",
    "O. sphegodes_Palena",
]


def list_images(directory):
    """List image files in a directory."""
    if not os.path.exists(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])


def md5(path):
    """Compute MD5 hash of a file."""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def split_class(source_dir, n_test, rng):
    """Split images in source_dir into test (n_test) and train (rest)."""
    files = list_images(source_dir)
    if len(files) < n_test:
        raise ValueError(
            f"Not enough images in {source_dir}: {len(files)} < {n_test}"
        )
    indices = rng.permutation(len(files))
    test_files = [files[i] for i in indices[:n_test]]
    train_files = [files[i] for i in indices[n_test:]]
    return train_files, test_files


def main():
    rng = np.random.default_rng(SEED)

    # Create output directories
    for cls in ALL_CLASSES:
        os.makedirs(TRAIN_CLEAN_DIR / cls, exist_ok=True)
        os.makedirs(TEST_CLEAN_DIR / cls, exist_ok=True)

    manifest = {
        "seed": SEED,
        "test_per_class": TEST_PER_CLASS,
        "timestamp": datetime.now().isoformat(),
        "classes": {},
    }

    total_train = 0
    total_test = 0

    for cls in ALL_CLASSES:
        cls_manifest = {"train_sources": {}, "test_source": None}

        # --- Test set ---
        if cls in INTERNET_CLASSES:
            # Test from internet_cropped
            source_dir = INTERNET_CROPPED_DIR / cls
            _, test_files = split_class(source_dir, TEST_PER_CLASS, rng)
            cls_manifest["test_source"] = f"internet_cropped/{cls}"
            cls_manifest["test_count"] = len(test_files)

            for f in test_files:
                shutil.copy2(source_dir / f, TEST_CLEAN_DIR / cls / f)

            # Train: all private + remaining internet
            train_files_priv = list_images(PRIVATE_DIR / cls)

            internet_all = list_images(source_dir)
            train_files_inet = [f for f in internet_all if f not in test_files]

            for f in train_files_priv:
                shutil.copy2(PRIVATE_DIR / cls / f, TRAIN_CLEAN_DIR / cls / f)
            for f in train_files_inet:
                shutil.copy2(source_dir / f, TRAIN_CLEAN_DIR / cls / f)

            cls_manifest["train_sources"] = {
                "private": len(train_files_priv),
                "internet_remaining": len(train_files_inet),
            }
            cls_manifest["train_count"] = len(train_files_priv) + len(train_files_inet)

        else:
            # Private-only class: test from private, rest to train
            source_dir = PRIVATE_DIR / cls
            train_files, test_files = split_class(source_dir, TEST_PER_CLASS, rng)
            cls_manifest["test_source"] = f"raw/{cls}"
            cls_manifest["test_count"] = len(test_files)

            for f in test_files:
                shutil.copy2(source_dir / f, TEST_CLEAN_DIR / cls / f)
            for f in train_files:
                shutil.copy2(source_dir / f, TRAIN_CLEAN_DIR / cls / f)

            cls_manifest["train_sources"] = {"private": len(train_files)}
            cls_manifest["train_count"] = len(train_files)

        total_train += cls_manifest["train_count"]
        total_test += cls_manifest["test_count"]
        manifest["classes"][cls] = cls_manifest

        print(f"{cls}:")
        print(f"  Train: {cls_manifest['train_count']} "
              f"({', '.join(f'{k}={v}' for k, v in cls_manifest['train_sources'].items())})")
        print(f"  Test:  {cls_manifest['test_count']} (from {cls_manifest['test_source']})")

    manifest["total_train"] = total_train
    manifest["total_test"] = total_test

    # Save manifest
    manifest_path = BASE_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
    print(f"Total: {total_train} train + {total_test} test = {total_train + total_test} images")


if __name__ == "__main__":
    main()
