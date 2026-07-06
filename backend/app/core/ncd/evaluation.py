"""OSR metriche: binary_novel AUROC + leave-one-out AUROC."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, roc_curve
from sklearn.metrics.cluster import adjusted_mutual_info_score, fowlkes_mallows_score, silhouette_score


def evaluate_novel(
    scores: NDArray,
    labels: NDArray,
    novel_name: str,
    class_names: list[str],
) -> dict:
    """AUROC + detection rate per novel class.

    novel_name matched contro class_names per nome (non per indice).
    Higher scores = more novel.
    AUROC < 0.5 NON auto-flippato: e' un risultato scientifico valido.
    detection_rate: massimo TPR a FPR <= 10%.
    """
    if novel_name not in class_names:
        return {
            "auroc": float("nan"),
            "detection_rate_at_10pct_fpr": float("nan"),
            "error": f"novel_name '{novel_name}' not in class_names {class_names}",
        }

    novel_label = class_names.index(novel_name)
    y_true = np.isin(labels, [novel_label]).astype(np.int64)

    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "detection_rate_at_10pct_fpr": float("nan")}

    auroc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= 0.1
    detection_rate = float(tpr[mask].max()) if mask.any() else 0.0

    return {"auroc": auroc, "detection_rate_at_10pct_fpr": detection_rate}


def evaluate_loo(
    scores: NDArray,
    labels: NDArray,
    class_names: list[str],
) -> dict:
    """Leave-one-class-out mean AUROC."""
    all_metrics = []
    for left_out in range(len(class_names)):
        y_true = np.isin(labels, [left_out]).astype(np.int64)

        if len(np.unique(y_true)) < 2:
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(y_true, scores))

        all_metrics.append({"auroc": auroc, "left_out": class_names[left_out]})

    valid = [m["auroc"] for m in all_metrics if not np.isnan(m["auroc"])]
    mean_auroc = float(np.mean(valid)) if valid else float("nan")

    return {"mean_auroc": mean_auroc, "per_class": all_metrics}


# ---- Clustering evaluation metrics (merged from metrics.py) ----


def hungarian_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute clustering accuracy using Hungarian algorithm (ACC).

    Maps each cluster to the best ground-truth class via linear assignment,
    then computes accuracy = correct / total.
    """
    assert true_labels.shape == pred_labels.shape
    true = true_labels.astype(np.int64)
    pred = pred_labels.astype(np.int64)
    n_classes = max(true.max(), pred.max()) + 1

    cost_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(len(true)):
        if pred[i] < 0:  # DBSCAN noise label
            continue
        cost_matrix[pred[i], true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    total_correct = cost_matrix[row_ind, col_ind].sum()
    return float(total_correct) / len(true)


def compute_nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Normalized Mutual Information."""
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def compute_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Rand Index."""
    return float(adjusted_rand_score(true_labels, pred_labels))


def compute_ami(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Mutual Information — corrects NMI for chance agreement."""
    return float(adjusted_mutual_info_score(true_labels, pred_labels))


def compute_f_measure(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """F-measure (Fowlkes-Mallows index): geometric mean of precision and recall."""
    return float(fowlkes_mallows_score(true_labels, pred_labels))


def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score — measures cluster cohesion vs separation (no labels needed)."""
    n_unique = len(set(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return -1.0
    return float(silhouette_score(embeddings, labels))


def compute_all_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    embeddings: np.ndarray | None = None,
) -> dict:
    result = {
        "acc": hungarian_accuracy(true_labels, pred_labels),
        "nmi": compute_nmi(true_labels, pred_labels),
        "ari": compute_ari(true_labels, pred_labels),
        "ami": compute_ami(true_labels, pred_labels),
        "f_measure": compute_f_measure(true_labels, pred_labels),
    }
    if embeddings is not None:
        result["silhouette"] = compute_silhouette(embeddings, pred_labels)
    return result
