"""OSR metriche: binary_novel AUROC + leave-one-out AUROC."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve


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
