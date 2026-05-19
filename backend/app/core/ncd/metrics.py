"""NCD evaluation metrics: ACC, NMI, ARI, AMI, F-measure, Silhouette."""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
)


def hungarian_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute clustering accuracy using Hungarian algorithm (ACC).

    Maps each cluster to the best ground-truth class via linear assignment,
    then computes accuracy = correct / total.
    """
    from scipy.optimize import linear_sum_assignment
    assert true_labels.shape == pred_labels.shape
    true = true_labels.astype(np.int64)
    pred = pred_labels.astype(np.int64)
    n_classes = max(true.max(), pred.max()) + 1

    cost_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(len(true)):
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
