"""Clean train/test split from cropped images.

Creates a deduplicated split with:
  - Internet classes: test from internet cropped, train from private + remaining internet
  - Private-only classes: test and train from private cropped

MD5 hashing ensures zero content overlap between train and test.
"""

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _md5(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _list_images(directory: str | Path) -> list[str]:
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def _split_class(source_dir: str | Path, n_test: int, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Split images into test (n_test) and train (rest)."""
    files = _list_images(source_dir)
    if len(files) < n_test:
        raise ValueError(f"Not enough images in {source_dir}: {len(files)} < {n_test}")
    indices = rng.permutation(len(files))
    test_files = [files[i] for i in indices[:n_test]]
    train_files = [files[i] for i in indices[n_test:]]
    return train_files, test_files


def create_clean_split(
    cropped_dir: str | Path,
    internet_dir: str | Path | None,
    output_dir: str | Path,
    class_names: list[str],
    internet_classes: Iterable[str],
    test_per_class: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a clean train/test split from cropped images.

    Args:
    ----
        cropped_dir: Cropped private images root (dataset/cropped/).
        internet_dir: Cropped internet images root (parent of {class}_Int/ dirs).
                      Can be None if all classes are private-only.
        output_dir: Output root. Creates train/ and test/ subdirs.
        class_names: List of all class names.
        internet_classes: Set/list of class names that have internet images.
        test_per_class: Number of test images per class (default 50).
        seed: Random seed for reproducibility.

    Returns:
    -------
        Manifest dict with split details.

    """
    cropped_dir = Path(cropped_dir)
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    internet_classes = set(internet_classes)

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    manifest: dict[str, Any] = {
        "seed": seed,
        "test_per_class": test_per_class,
        "timestamp": datetime.now().isoformat(),
        "classes": {},
    }

    total_train = 0
    total_test = 0

    for cls in class_names:
        train_cls = train_dir / cls
        test_cls = test_dir / cls
        train_cls.mkdir(parents=True, exist_ok=True)
        test_cls.mkdir(parents=True, exist_ok=True)

        cls_manifest: dict[str, Any] = {"train_sources": {}, "test_source": None}

        if cls in internet_classes and internet_dir:
            inet_dir = Path(internet_dir)
            # Find the matching internet directory by normalized name
            cls_normalized = cls.replace(" ", "")
            inet_class_dir = None
            for candidate in inet_dir.iterdir():
                if not candidate.is_dir():
                    continue
                # Strip common suffixes: "O.exaltata_Int" or "O.exaltata" → "O.exaltata"
                cand_name = candidate.name.replace("_Int", "")
                if cand_name == cls_normalized:
                    inet_class_dir = candidate
                    break

            if inet_class_dir is None:
                raise FileNotFoundError(
                    f"Internet directory not found for class '{cls}'. "
                    f"Expected something like {inet_dir}/{cls.replace(' ', '')}_Int/"
                )

            # Test from internet
            _, test_files = _split_class(inet_class_dir, test_per_class, rng)
            cls_manifest["test_source"] = str(inet_class_dir.relative_to(inet_dir.parent))

            # Build set of test content hashes for dedup
            test_hashes: set[str] = set()
            for f in test_files:
                shutil.copy2(inet_class_dir / f, test_cls / f)
                test_hashes.add(_md5(test_cls / f))

            # Train: all private cropped + remaining internet (deduped by MD5)
            train_private = _list_images(cropped_dir / cls)
            inet_all = _list_images(inet_class_dir)
            train_inet = [f for f in inet_all if f not in test_files]

            skipped_priv = 0
            skipped_inet = 0
            for f in train_private:
                src = cropped_dir / cls / f
                if _md5(src) in test_hashes:
                    skipped_priv += 1
                    continue
                shutil.copy2(src, train_cls / f)
            for f in train_inet:
                src = inet_class_dir / f
                if _md5(src) in test_hashes:
                    skipped_inet += 1
                    continue
                shutil.copy2(src, train_cls / f)

            private_kept = len(train_private) - skipped_priv
            inet_kept = len(train_inet) - skipped_inet
            cls_manifest["train_sources"] = {
                "private_cropped": private_kept,
                "internet_remaining": inet_kept,
                "skipped_md5_dupes": skipped_priv + skipped_inet,
            }
            cls_manifest["train_count"] = private_kept + inet_kept
            cls_manifest["test_count"] = len(test_files)

        else:
            # Private-only: split from private cropped
            source = cropped_dir / cls
            train_files, test_files = _split_class(source, test_per_class, rng)
            cls_manifest["test_source"] = f"cropped/{cls}"

            for f in test_files:
                shutil.copy2(source / f, test_cls / f)
            for f in train_files:
                shutil.copy2(source / f, train_cls / f)

            cls_manifest["train_sources"] = {"private_cropped": len(train_files)}
            cls_manifest["train_count"] = len(train_files)
            cls_manifest["test_count"] = len(test_files)

        total_train += cls_manifest["train_count"]
        total_test += cls_manifest["test_count"]
        manifest["classes"][cls] = cls_manifest

        logger.info(
            f"{cls}: train={cls_manifest['train_count']}, "
            f"test={cls_manifest['test_count']} "
            f"(from {cls_manifest['test_source']})"
        )

    manifest["total_train"] = total_train
    manifest["total_test"] = total_test

    # MD5 dedup check
    train_hashes = set()
    test_hashes = set()
    for cls in class_names:
        for f in _list_images(train_dir / cls):
            train_hashes.add(_md5(train_dir / cls / f))
        for f in _list_images(test_dir / cls):
            h = _md5(test_dir / cls / f)
            if h in train_hashes:
                logger.warning(f"DATA LEAK: {cls}/{f} in both train and test!")
            test_hashes.add(h)

    overlap = train_hashes & test_hashes
    manifest["md5_overlap"] = len(overlap)
    if overlap:
        logger.error(f"MD5 overlap: {len(overlap)} images in both train and test!")
    else:
        logger.info("MD5 check: zero overlap between train and test")

    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Manifest saved: {manifest_path}")
    logger.info(f"Total: {total_train} train + {total_test} test = {total_train + total_test} images")

    return manifest
