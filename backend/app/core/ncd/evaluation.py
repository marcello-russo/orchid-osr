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


def evaluate_osr(
    scores: NDArray,
    labels: NDArray,
    class_names: list[str],
    exclude_classes: list[str],
) -> dict:
    """OSR evaluation con supporto 1 o 2 novel.

    Se len(exclude_classes) == 1: AUROC globale unico (compatibile con evaluate_novel).
    Se len(exclude_classes) == 2: calcola 3 AUROC:
        - auroc globale: known (200) vs novel_combined (100)
        - auroc_per_novel: dict con AUROC per ciascuna novel separata (known vs singola novel)

    Ritorna dict con chiavi "auroc", "auroc_per_novel" (se 2 novel), "detection_rate_at_10pct_fpr".
    """
    if not exclude_classes:
        raise ValueError("exclude_classes vuoto")

    for c in exclude_classes:
        if c not in class_names:
            return {
                "auroc": float("nan"),
                "auroc_per_novel": {},
                "error": f"exclude class '{c}' not in class_names {class_names}",
            }

    novel_indices = [class_names.index(c) for c in exclude_classes]

    # AUROC globale: known vs novel_combined
    y_true_global = np.isin(labels, novel_indices).astype(np.int64)
    if len(np.unique(y_true_global)) < 2:
        auroc_global = float("nan")
        detection_rate = float("nan")
    else:
        auroc_global = float(roc_auc_score(y_true_global, scores))
        fpr, tpr, _ = roc_curve(y_true_global, scores)
        mask = fpr <= 0.1
        detection_rate = float(tpr[mask].max()) if mask.any() else 0.0

    if len(exclude_classes) == 1:
        return {"auroc": auroc_global, "detection_rate_at_10pct_fpr": detection_rate}

    # 2 novel: calcola per-novel
    auroc_per_novel: dict[str, float] = {}
    for novel_cls, novel_idx in zip(exclude_classes, novel_indices):
        # Filtra solo known + questa novel (esclude l'altra novel)
        known_indices = [i for i, n in enumerate(class_names) if n not in exclude_classes]
        # Labels considerati: known vs questa novel
        mask_keep = np.isin(labels, known_indices + [novel_idx])
        if mask_keep.sum() == 0 or len(np.unique(labels[mask_keep])) < 2:
            auroc_per_novel[novel_cls] = float("nan")
            continue
        y_true = (labels[mask_keep] == novel_idx).astype(np.int64)
        filtered_scores = scores[mask_keep]
        if len(np.unique(y_true)) < 2:
            auroc_per_novel[novel_cls] = float("nan")
        else:
            auroc_per_novel[novel_cls] = float(roc_auc_score(y_true, filtered_scores))

    return {
        "auroc": auroc_global,
        "auroc_per_novel": auroc_per_novel,
        "detection_rate_at_10pct_fpr": detection_rate,
    }


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


def hungarian_mapping(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[int, int]:
    """Mapping cluster -> classe vera via assegnazione ungherese (max match)."""
    true = true_labels.astype(np.int64)
    pred = pred_labels.astype(np.int64)
    n_classes = max(true.max(), pred.max()) + 1
    cost_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(len(true)):
        if pred[i] < 0:
            continue
        cost_matrix[pred[i], true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


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


def gcd_old_new_all_acc(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    known_mask: np.ndarray,
) -> dict:
    """ACC Old / New / All secondo il protocollo GCD (Vaze et al., CVPR 2022).

    Il mapping cluster->classe e' risolto UNA volta con l'assegnazione ungherese
    sull'intero pool, poi l'accuratezza e' riportata sui sottoinsiemi:
    - old: campioni delle classi note (known_mask True)
    - new: campioni delle classi novel (known_mask False)
    - all: intero pool
    """
    assert true_labels.shape == pred_labels.shape
    true = true_labels.astype(np.int64)
    pred = pred_labels.astype(np.int64)
    n_classes = max(true.max(), pred.max()) + 1

    cost_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(len(true)):
        if pred[i] < 0:
            continue
        cost_matrix[pred[i], true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    cluster_to_class = dict(zip(row_ind.tolist(), col_ind.tolist()))

    mapped = np.array([cluster_to_class.get(int(p), -1) for p in pred])
    correct = mapped == true

    known_mask = np.asarray(known_mask, dtype=bool)
    n_old = int(known_mask.sum())
    n_new = int((~known_mask).sum())
    return {
        "acc_all": float(correct.mean()) if len(true) else float("nan"),
        "acc_old": float(correct[known_mask].mean()) if n_old else float("nan"),
        "acc_new": float(correct[~known_mask].mean()) if n_new else float("nan"),
        "n_old": n_old,
        "n_new": n_new,
    }


def per_novel_acc(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pool_species: np.ndarray,
    novel_names: list[str],
) -> dict:
    """ACC di scoperta separata per ciascuna specie novel.

    Usa lo stesso mapping cluster->classe risolto globalmente (Hungarian):
    per ogni specie novel riporta la frazione dei suoi campioni assegnati
    al cluster mappato su quella specie.
    """
    true = true_labels.astype(np.int64)
    pred = pred_labels.astype(np.int64)
    mapping = hungarian_mapping(true, pred)
    out: dict[str, float] = {}
    for sp in novel_names:
        mask = pool_species == sp
        if int(mask.sum()) == 0:
            out[sp] = float("nan")
            continue
        mapped = np.array([mapping.get(int(p), -1) for p in pred[mask]])
        true_idx = int(true[mask][0])
        out[sp] = float((mapped == true_idx).mean())
    return out


def pool_hungarian_test_acc(
    pool_true: np.ndarray,
    pred: np.ndarray,
    n_train: int,
    test_known_mask: np.ndarray,
) -> dict:
    """ACC test da mapping Hungarian fittato sul pool (verbatim gcd_analysis.py:180-189).

    Il mapping cluster->classe è risolto UNA volta sul pool completo
    (labeled + unlabeled), poi le accuratezze old/new/all sono valutate
    sulle sole ``n_test`` del test (slice posizionale dopo ``n_train``).
    NON usare ``gcd_old_new_all_acc`` sul solo test: mapping diverso,
    non confrontabile (piano impl §2).
    """
    mapping = hungarian_mapping(np.asarray(pool_true), np.asarray(pred))
    idx_test = np.arange(n_train, n_train + len(np.asarray(test_known_mask)))
    mapped_test = np.array([mapping.get(int(p), -1) for p in np.asarray(pred)[idx_test]])
    correct_test = mapped_test == np.asarray(pool_true)[idx_test]
    mask_new_test = ~np.asarray(test_known_mask, dtype=bool)
    mapped_pool = np.array([mapping.get(int(p), -1) for p in np.asarray(pred)])
    return {
        "acc_all_test": float(correct_test.mean()),
        "acc_old_test": (
            float(correct_test[np.asarray(test_known_mask, dtype=bool)].mean())
            if np.asarray(test_known_mask).sum()
            else float("nan")
        ),
        "acc_new_test": float(correct_test[mask_new_test].mean()) if mask_new_test.sum() else float("nan"),
        "acc_all_pool": float((mapped_pool == np.asarray(pool_true)).mean()),
        "n_old_test": int(np.asarray(test_known_mask).sum()),
        "n_new_test": int(mask_new_test.sum()),
    }


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
