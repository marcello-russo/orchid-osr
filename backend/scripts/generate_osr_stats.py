"""Generate OSR calibration stats for all models with ncd_mode.

For each model, loads saved embeddings, computes detector statistics
(Mahalanobis + KNN), performs calibration at 10% FPR using z-score
normalization (consistent with runtime), and saves osr_stats.npz.

Usage: python scripts/generate_osr_stats.py
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.cli.definitions import EXPERIMENTS as _EXPERIMENTS
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = settings.class_names
N_CLASSES = len(CLASS_NAMES)
OUT_DIR = Path("data/osr_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_mahalanobis(embs: np.ndarray, means: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    scores = np.full(len(embs), np.inf)
    for i in range(N_CLASSES):
        diff = embs - means[i]
        dist = np.sum(diff @ cov_inv * diff, axis=1)
        scores = np.minimum(scores, dist)
    return scores


def compute_knn(embs: np.ndarray, train: np.ndarray, k: int = 5) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k, len(train)), metric="euclidean", n_jobs=-1)
    nn.fit(train)
    dists, _ = nn.kneighbors(embs)
    return dists.mean(axis=1)


def calibrate_zscore(scores_known: np.ndarray, scores_novel: np.ndarray, fpr: float = 0.1):
    """Find threshold at given FPR and compute detection rate.

    scores are already z-scored: lower = more known, higher = more novel.
    Known samples have lower scores, novel have higher scores.
    Threshold: the highest score that rejects (1-fpr) of novel = novel score at fpr quantile.
    """
    sorted_novel = np.sort(scores_novel)
    idx = max(0, int(len(scores_novel) * fpr) - 1)
    threshold = float(sorted_novel[idx])
    detected = float(np.mean(scores_known <= threshold))
    return threshold, detected


def process(model_name: str) -> dict | None:
    exp_dir = Path(settings.results_dir)
    model_dir = exp_dir / model_name

    train_dir = model_dir / "embeddings"
    if not train_dir.is_dir() or not (train_dir / "embeddings.npy").is_file():
        logger.warning("  SKIP %s: no train embeddings", model_name)
        return None
    if not (train_dir / "labels.npy").is_file():
        logger.warning("  SKIP %s: no labels.npy", model_name)
        return None

    train_embs = np.load(train_dir / "embeddings.npy")
    train_labels = np.load(train_dir / "labels.npy")
    logger.info("  %s: train=%s", model_name, train_embs.shape)

    # Test embeddings with novel class (for calibration)
    novel_dir = model_dir / "embeddings_with_novel"
    has_cal = novel_dir.is_dir()
    if has_cal:
        test_embs = np.load(novel_dir / "embeddings.npy")
        test_labels = np.load(novel_dir / "labels.npy")
        known_mask = test_labels < N_CLASSES
        novel_mask = test_labels >= N_CLASSES
        known_embs = test_embs[known_mask]
        novel_embs = test_embs[novel_mask]
        logger.info("    calibration: known=%d novel=%d", len(known_embs), len(novel_embs))
    else:
        known_embs = np.empty((0, train_embs.shape[1]))
        novel_embs = np.empty((0, train_embs.shape[1]))

    # Class means
    means = np.zeros((N_CLASSES, train_embs.shape[1]), dtype=np.float32)
    for i in range(N_CLASSES):
        mask = train_labels == i
        if mask.sum():
            means[i] = train_embs[mask].mean(axis=0)

    # Pooled covariance
    cov = np.zeros((train_embs.shape[1], train_embs.shape[1]), dtype=np.float64)
    for i in range(N_CLASSES):
        mask = train_labels == i
        if mask.sum() > 1:
            centered = train_embs[mask] - means[i]
            cov += centered.T @ centered
    cov /= len(train_labels)
    cov_inv = np.linalg.pinv(cov).astype(np.float32)

    # Training set scores (for z-score normalization params)
    train_maha = compute_mahalanobis(train_embs, means, cov_inv)
    train_knn = compute_knn(train_embs, train_embs)
    maha_mean, maha_std = float(train_maha.mean()), float(train_maha.std())
    knn_mean, knn_std = float(train_knn.mean()), float(train_knn.std())

    # Calibrate Ensemble threshold at 10% FPR (Mahalanobis + KNN, z-score normalized)
    threshold, det_rate = 0.5, 0.0
    if len(novel_embs) > 0:
        known_maha = compute_mahalanobis(known_embs, means, cov_inv)
        novel_maha = compute_mahalanobis(novel_embs, means, cov_inv)
        known_knn = compute_knn(known_embs, train_embs)
        novel_knn = compute_knn(novel_embs, train_embs)

        # Z-score normalize using TRAINING set statistics (consistent with runtime)
        known_maha_z = (known_maha - maha_mean) / max(maha_std, 1e-8)
        novel_maha_z = (novel_maha - maha_mean) / max(maha_std, 1e-8)
        known_knn_z = (known_knn - knn_mean) / max(knn_std, 1e-8)
        novel_knn_z = (novel_knn - knn_mean) / max(knn_std, 1e-8)

        known_ens = (known_maha_z + known_knn_z) / 2
        novel_ens = (novel_maha_z + novel_knn_z) / 2
        threshold, det_rate = calibrate_zscore(known_ens, novel_ens)
    else:
        logger.warning("    No calibration data, default threshold=0.5")

    # Save
    out = OUT_DIR / f"{model_name}.npz"
    np.savez_compressed(
        out,
        class_means=means,
        cov_inv=cov_inv,
        train_embeddings=train_embs.astype(np.float32),
        threshold_ensemble=np.array([threshold], dtype=np.float32),
        detection_rate=np.array([det_rate], dtype=np.float32),
        maha_mean=np.array([maha_mean], dtype=np.float32),
        maha_std=np.array([maha_std], dtype=np.float32),
        knn_mean=np.array([knn_mean], dtype=np.float32),
        knn_std=np.array([knn_std], dtype=np.float32),
    )
    size_mb = out.stat().st_size / 1e6
    logger.info("    => %.1fMB DR@10%%=%.3f thresh=%.4f", size_mb, det_rate, threshold)
    return {"model": model_name, "detection_rate": det_rate, "size_mb": size_mb}


def main():
    exp_dir = Path(settings.results_dir)
    results = []

    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir() or "old_hf" in d.name:
            continue
        name = d.name
        exp_num = next(
            (n for p, n in [("exp6", 6), ("exp7", 7), ("exp8", 8), ("exp9", 9), ("exp10", 10)] if p in name), None
        )
        if exp_num is None or not _EXPERIMENTS.get(exp_num, {}).get("ncd_mode", False):
            continue
        if not (d / "embeddings").is_dir():
            logger.info("  SKIP %s (no embeddings dir)", name)
            continue

        r = process(name)
        if r:
            results.append(r)

    total_mb = sum(r["size_mb"] for r in results)
    logger.info("\n=== Done: %d models, %.1fMB total ===", len(results), total_mb)
    for r in results:
        logger.info("  %s: DR=%.3f  %.1fMB", r["model"], r["detection_rate"], r["size_mb"])


if __name__ == "__main__":
    main()
