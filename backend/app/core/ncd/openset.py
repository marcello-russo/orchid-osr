"""Open-Set Recognition: OpenMax (EVT-based) + OSR metrics."""

import logging

import numpy as np
import torch
from scipy.stats import exponweib

logger = logging.getLogger(__name__)


def fit_weibull(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    tail_size: int = 20,
) -> dict:
    """Fit Weibull distribution for each known class.

    For each class, computes distances of training samples to the class mean,
    then fits a Weibull distribution to the tail (top-k largest distances).

    Args:
        embeddings: (N, D) array of L2-normalized embeddings.
        labels: (N,) array of integer labels.
        class_names: list of class names.
        tail_size: number of extreme distances to fit Weibull (default 20).

    Returns:
        dict mapping class_idx -> {
            "mean": centroid vector,
            "weibull_shape": float,
            "weibull_scale": float,
        }
    """
    from scipy.stats import exponweib

    class_models = {}
    unique_labels = sorted(set(labels))

    for cls in unique_labels:
        mask = labels == cls
        cls_emb = embeddings[mask]
        centroid = cls_emb.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)

        dists = np.linalg.norm(cls_emb - centroid_norm, axis=1)

        sorted_dists = np.sort(dists)[-tail_size:]
        if len(sorted_dists) < 2:
            sorted_dists = np.array([0.01, 0.02])

        params = exponweib.fit(sorted_dists, floc=0, f0=1)
        shape, _, _, scale = params

        class_models[int(cls)] = {
            "mean": centroid_norm,
            "weibull_shape": float(shape),
            "weibull_scale": float(scale),
            "class_name": class_names[cls] if cls < len(class_names) else str(cls),
        }

    return class_models


def compute_openmax_scores(
    embeddings: np.ndarray,
    class_models: dict,
    num_classes: int,
    alpha: int = 4,
) -> np.ndarray:
    """Compute OpenMax scores: probability of belonging to each known class + unknown.

    For each sample, computes logit-like activation scores to each known class.
    The scores are Weibull-CDF reweighted: the top-alpha activations are
    reduced based on their Weibull probability, and the remainder is assigned
    to the 'unknown' class.

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        class_models: dict from fit_weibull().
        num_classes: number of known classes.
        alpha: number of top classes to revise (default 4).

    Returns:
        scores: (N, num_classes + 1) where the last column is 'unknown' score.
    """
    N = embeddings.shape[0]
    scores = np.zeros((N, num_classes + 1))

    for i in range(N):
        emb = embeddings[i]
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        activations = []
        for cls in range(num_classes):
            if cls not in class_models:
                activations.append(0.0)
                continue
            centroid = class_models[cls]["mean"]
            sim = float(np.dot(emb, centroid))
            activations.append(sim)

        activations = np.array(activations)

        sorted_idx = np.argsort(activations)[::-1]
        top_alpha = sorted_idx[:alpha]

        revised = activations.copy()
        for cls in top_alpha:
            cm = class_models[int(cls)]
            dist = np.linalg.norm(emb - cm["mean"])
            wscore = exponweib.cdf(dist, 1, cm["weibull_shape"], loc=0, scale=cm["weibull_scale"])
            revised[cls] = activations[cls] * (1 - wscore)

        denom = np.sum(revised)
        if denom > 0:
            scores[i, :num_classes] = revised / denom
            scores[i, num_classes] = 1.0 - np.max(revised / denom)
        else:
            scores[i, :] = 0.0
            scores[i, num_classes] = 1.0

    return scores


def baseline_softmax_scores(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute softmax-based OSR scores.

    Unknown class gets (1 - max softmax probability).

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        prototypes: (K, D) L2-normalized class prototypes.
        temperature: scaling factor (higher = softer distribution).

    Returns:
        scores: (N, K+1) where last column is unknown score.
    """
    sim = np.dot(embeddings, prototypes.T)
    scaled = sim / temperature
    exp_sim = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    softmax = exp_sim / exp_sim.sum(axis=1, keepdims=True)

    unknown = 1.0 - softmax.max(axis=1)
    scores = np.column_stack([softmax, unknown])
    return scores


def compute_oscr_curve(
    reject_rates: np.ndarray,
    known_accs: np.ndarray,
) -> list[dict]:
    """Build OSCR curve points as list of {reject_rate, known_acc}."""
    points = []
    for rr, ka in zip(reject_rates, known_accs):
        points.append({"reject_rate": round(float(rr), 4), "known_acc": round(float(ka), 4)})
    return points


def compute_oscr_from_scores(
    scores: np.ndarray,
    true_labels: np.ndarray,
    known_classes: list[int],
    n_thresholds: int = 100,
) -> list[dict]:
    """Compute OSCR curve points from model scores by varying unknown rejection threshold."""
    unknown_score = scores[:, -1]
    is_known_true = np.isin(true_labels, known_classes).astype(float)
    valid_known = is_known_true == 1

    thresholds = np.linspace(0, 1, n_thresholds)
    known_accs = []
    reject_rates = []

    for thresh in thresholds:
        is_known_pred = unknown_score < thresh
        reject_rates.append(float((~is_known_pred).mean()))
        if valid_known.any():
            known_accs.append(float((is_known_pred[valid_known] == is_known_true[valid_known]).mean()))
        else:
            known_accs.append(0.0)

    return compute_oscr_curve(np.array(reject_rates), np.array(known_accs))


def compute_osr_metrics(
    scores: np.ndarray,
    true_labels: np.ndarray,
    known_classes: list[int],
) -> dict:
    """Compute Open-Set Recognition metrics.

    Args:
        scores: (N, num_classes + 1) from compute_openmax_scores.
        true_labels: (N,) true integer labels (-1 for unknown).
        known_classes: list of known class indices.

    Returns:
        dict with AUROC, OSCR, accuracy, unknown_recall.
    """
    from sklearn.metrics import roc_auc_score

    pred_labels = scores.argmax(axis=1)
    is_known_true = np.isin(true_labels, known_classes).astype(float)
    is_known_pred = (pred_labels < scores.shape[1] - 1).astype(float)

    unknown_score = scores[:, -1]

    valid_known = is_known_true == 1
    valid_unknown = is_known_true == 0
    if valid_known.any() and valid_unknown.any():
        auroc = float(roc_auc_score(is_known_true, -unknown_score))
    else:
        auroc = float("nan")

    correct = (pred_labels == true_labels) & (is_known_true == 1)
    known_acc = float(correct[valid_known].mean()) if valid_known.any() else float("nan")

    unknown_recall = float((is_known_pred[valid_unknown] == 0).mean()) if valid_unknown.any() else float("nan")

    return {
        "auroc": auroc,
        "known_accuracy": known_acc,
        "unknown_recall": unknown_recall,
        "num_known": int(valid_known.sum()),
        "num_unknown": int(valid_unknown.sum()),
    }


def run_openset_evaluation(
    exp_dir: str,
    model_variant: str,
    num_classes: int,
    class_names: list[str],
    train_dir: str,
    test_dir: str,
    batch_size: int = 8,
) -> dict:
    """Run full Open-Set evaluation: extract embeddings, fit Weibull, compute OpenMax.

    All known classes are used for training. A held-out subset of test images
    is treated as 'unknown' to measure OSR performance.

    Returns:
        dict with OSR metrics.
    """
    import json
    import os

    from app.core.ncd.extract import extract_embeddings

    results_path = os.path.join(exp_dir, "results.json")
    if not os.path.exists(results_path):
        logger.error(f"results.json not found in {exp_dir}")
        return {}

    with open(results_path) as f:
        results = json.load(f)

    best_fold = results.get("best_fold", 0)
    best_model_path = os.path.join(exp_dir, f"best_fold_{best_fold}.pt")
    if not os.path.exists(best_model_path):
        logger.error(f"Best model not found: {best_model_path}")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model {model_variant} (best fold {best_fold})...")

    train_emb, train_labels, _, train_species = extract_embeddings(
        model_variant=model_variant,
        data_dir=train_dir,
        best_model_path=best_model_path,
        num_classes=num_classes,
        batch_size=batch_size,
        device=device,
    )

    test_emb, test_labels, _, test_species = extract_embeddings(
        model_variant=model_variant,
        data_dir=test_dir,
        best_model_path=best_model_path,
        num_classes=num_classes,
        batch_size=batch_size,
        device=device,
    )

    logger.info(f"Train embeddings: {train_emb.shape}, Test embeddings: {test_emb.shape}")

    # Normalize
    train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-12)
    test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-12)

    # Fit Weibull on known classes
    class_models = fit_weibull(train_norm, train_labels, class_names)

    # Compute OpenMax scores
    scores = compute_openmax_scores(test_norm, class_models, num_classes)

    # Baseline: softmax scores
    prototypes = np.array([class_models[c]["mean"] for c in sorted(class_models.keys())])
    baseline_scores = baseline_softmax_scores(test_norm, prototypes)

    # Compute OSCR curves
    openmax_oscr = compute_oscr_from_scores(scores, test_labels, list(range(num_classes)))
    baseline_oscr = compute_oscr_from_scores(baseline_scores, test_labels, list(range(num_classes)))

    # OSR metrics: simulate unknown by treating each class as unknown in turn
    all_metrics = []
    unknown_labels = np.full(len(test_labels), -1)

    # Leave-one-class-out evaluation
    for left_out in range(num_classes):
        val_labels = unknown_labels.copy()
        val_labels[test_labels != left_out] = test_labels[test_labels != left_out]
        val_known = [c for c in range(num_classes) if c != left_out]

        metrics = compute_osr_metrics(scores, val_labels, val_known)
        metrics["left_out"] = class_names[left_out]
        all_metrics.append(metrics)
        logger.info(
            f"  Left out {class_names[left_out]:20s}: "
            f"AUROC={metrics['auroc']:.4f}  "
            f"UnknownRecall={metrics['unknown_recall']:.4f}"
        )

    mean_auroc = np.mean([m["auroc"] for m in all_metrics])
    logger.info(f"\nMean AUROC (leave-one-class-out): {mean_auroc:.4f}")

    # Baseline AUROC
    baseline_auroc_list = []
    for left_out in range(num_classes):
        val_labels = unknown_labels.copy()
        val_labels[test_labels != left_out] = test_labels[test_labels != left_out]
        val_known = [c for c in range(num_classes) if c != left_out]
        base_metrics = compute_osr_metrics(baseline_scores, val_labels, val_known)
        baseline_auroc_list.append(base_metrics["auroc"])

    result = {
        "model_variant": model_variant,
        "num_classes": num_classes,
        "class_names": class_names,
        "mean_auroc": float(mean_auroc),
        "per_class_metrics": all_metrics,
        "oscr_curve": openmax_oscr,
        "baseline_oscr_curve": baseline_oscr,
        "baselines": {
            "softmax_threshold": {
                "mean_auroc": float(np.mean(baseline_auroc_list)),
            }
        },
    }

    osr_path = os.path.join(exp_dir, "osr_results.json")
    with open(osr_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"OSR results saved to: {osr_path}")

    return result


# ──────────────────────────────────────────────
# Ensemble OSR scoring (Mahalanobis + KNN + Energy)
# ──────────────────────────────────────────────


def _estimate_precision(embeddings: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Compute regularized precision matrix (inverse covariance)."""
    cov = np.cov(embeddings, rowvar=False)
    cov += reg * np.eye(cov.shape[0])
    return np.linalg.inv(cov)


def compute_mahalanobis_scores(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Mahalanobis distance-based OOD scores. Higher = more unknown."""
    train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-12)
    test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-12)

    unknown_scores = np.zeros(len(test_emb))
    for c in range(num_classes):
        c_emb = train_norm[train_labels == c]
        if len(c_emb) < 2:
            continue
        mean = c_emb.mean(axis=0)
        precision = _estimate_precision(c_emb)
        diff = test_norm - mean
        maha = np.sum(diff @ precision * diff, axis=1)
        unknown_scores = np.maximum(unknown_scores, maha)

    # Normalize to [0, 1]
    return unknown_scores / (unknown_scores.max() + 1e-12)


def compute_knn_scores(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    k: int = 5,
    k_bg: int = 50,
) -> np.ndarray:
    """KNN density ratio scores. Higher = farther from known = more unknown."""
    from sklearn.neighbors import NearestNeighbors

    train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-12)
    test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-12)

    nn_known = NearestNeighbors(n_neighbors=min(k, len(train_norm)))
    nn_known.fit(train_norm)
    d_known, _ = nn_known.kneighbors(test_norm)

    nn_bg = NearestNeighbors(n_neighbors=min(k_bg, len(train_norm)))
    nn_bg.fit(train_norm)
    d_bg, _ = nn_bg.kneighbors(test_norm)

    scores = d_known.mean(axis=1) / (d_bg.mean(axis=1) + 1e-12)
    return scores / (scores.max() + 1e-12)


def compute_energy_scores(
    logits: np.ndarray,
) -> np.ndarray:
    """Energy-based OOD scores. Lower energy = more unknown.
    Score is inverted so higher = more unknown (consistent with other methods).
    """
    energy = np.log(np.sum(np.exp(logits), axis=1))
    scores = -energy
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def compute_ensemble_scores(
    maha_scores: np.ndarray,
    knn_scores: np.ndarray,
    energy_scores: np.ndarray,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> np.ndarray:
    """Weighted ensemble of three OSR scores. Higher = more unknown."""
    return weights[0] * maha_scores + weights[1] * knn_scores + weights[2] * energy_scores


# ──────────────────────────────────────────────
# Ensemble OSR detector class
# ──────────────────────────────────────────────


class _MahalanobisDetector:
    """Mahalanobis distance-based OOD detector."""

    def fit(self, embeddings: np.ndarray, reg: float = 1e-6):
        cov = np.cov(embeddings, rowvar=False)
        cov += reg * np.eye(cov.shape[0])
        self.mean_ = embeddings.mean(axis=0)
        self.precision_ = np.linalg.inv(cov)

    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        diff = embeddings - self.mean_
        return np.sum(diff @ self.precision_ * diff, axis=1)


class _KNNDetector:
    """KNN distance-based OOD detector."""

    def fit(self, embeddings: np.ndarray, k: int = 5):
        from sklearn.neighbors import NearestNeighbors

        self.k = min(k, len(embeddings))
        self.nn_ = NearestNeighbors(n_neighbors=self.k)
        self.nn_.fit(embeddings)

    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        distances, _ = self.nn_.kneighbors(embeddings)
        return distances.mean(axis=1)


class EnsembleOSR:
    """Ensemble OSR detector combining Mahalanobis, KNN, and (optionally) Energy.

    Attributes:
        mahalanobis_dist: _MahalanobisDetector instance (or None if not fitted).
        knn_dist: _KNNDetector instance (or None if not fitted).
        energy_model: reserved for future Energy-based detector (always None for now).
    """

    def __init__(self):
        self.mahalanobis_dist: _MahalanobisDetector | None = None
        self.knn_dist: _KNNDetector | None = None
        self.energy_model = None

    def fit(self, embeddings: np.ndarray):
        """Fit Mahalanobis and KNN detectors on known-class embeddings."""
        self.mahalanobis_dist = _MahalanobisDetector()
        self.mahalanobis_dist.fit(embeddings)

        self.knn_dist = _KNNDetector()
        self.knn_dist.fit(embeddings)

    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        """Return novelty scores for each embedding (higher = more novel)."""
        scores = np.zeros(len(embeddings))
        count = 0
        if self.mahalanobis_dist is not None:
            mahal = self.mahalanobis_dist.score_samples(embeddings)
            scores += self._normalize(mahal)
            count += 1
        if self.knn_dist is not None:
            knn = self.knn_dist.score_samples(embeddings)
            scores += self._normalize(knn)
            count += 1
        if hasattr(self, "energy_model") and self.energy_model is not None:
            energy = self.energy_model.score_samples(embeddings)
            scores += self._normalize(energy)
            count += 1
        return scores / max(count, 1)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-10:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)


def evaluate_osr_novel(
    model,
    train_loader,
    test_loader,
    novel_label,
    known_labels,
    device="cuda",
):
    """Evaluate OSR detection of a single novel class.

    Args:
        model: Trained model (backbone + projector). Has .get_embeddings() method.
        train_loader: Training set (known classes only).
        test_loader: Test set (known + novel classes).
        novel_label: Label index of the novel class.
        known_labels: List of label indices for known classes.
        device: torch device.

    Returns:
        dict with keys: auroc, detection_rate_at_01_fpr, tpr, fpr, thresholds, num_known, num_novel
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    model.eval()
    model.to(device)

    train_embeddings = []
    for inputs, labels, _ in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.no_grad():
            emb = model.get_embeddings(inputs)
        train_embeddings.append(emb.cpu().numpy())
    train_embeddings = np.concatenate(train_embeddings, axis=0)

    test_embeddings = []
    test_labels = []
    for inputs, labels, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.no_grad():
            emb = model.get_embeddings(inputs)
        test_embeddings.append(emb.cpu().numpy())
        test_labels.append(labels.cpu().numpy())
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    detector = EnsembleOSR()
    detector.fit(train_embeddings)

    scores = detector.score_samples(test_embeddings)

    y_true = np.isin(test_labels, [novel_label]).astype(np.int64)

    auroc = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    mask = fpr <= 0.1
    detection_rate = float(tpr[mask].max()) if mask.any() else 0.0

    return {
        "auroc": float(auroc),
        "detection_rate_at_01_fpr": detection_rate,
        "tpr": tpr.tolist(),
        "fpr": fpr.tolist(),
        "thresholds": thresholds.tolist(),
        "num_known": int((y_true == 0).sum()),
        "num_novel": int((y_true == 1).sum()),
    }
