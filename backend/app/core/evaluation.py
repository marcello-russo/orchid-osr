# app/core/evaluation.py
import logging
import warnings
from typing import Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix as sk_confusion_matrix,
)

from app.core.models.base import ClassifierModel

logger = logging.getLogger(__name__)


def evaluate(
    model: ClassifierModel,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to use

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - f1_macro: Macro F1 score
            - precision_macro: Macro precision
            - recall_macro: Macro recall
            - f1_per_class: F1 score per class (dict)
            - confusion_matrix: Confusion matrix as 2D list
            - per_class_metrics: Detailed metrics per class (dict)
    """
    model.eval()
    num_classes = len(class_names)

    # Logga warning sklearn una sola volta, non per ogni metrica
    warnings.filterwarnings("once", message="Some classes do not exist")
    warnings.filterwarnings("once", message="One or more NaNs identified")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute macro metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0.0)
    f1_micro = f1_score(all_labels, all_predictions, average="micro", zero_division=0.0)
    precision_macro = precision_score(all_labels, all_predictions, average="macro", zero_division=0.0)
    recall_macro = recall_score(all_labels, all_predictions, average="macro", zero_division=0.0)

    # Compute per-class metrics
    f1_per_class_values = f1_score(all_labels, all_predictions, average=None, zero_division=0.0, labels=list(range(num_classes)))
    precision_per_class_values = precision_score(all_labels, all_predictions, average=None, zero_division=0.0, labels=list(range(num_classes)))
    recall_per_class_values = recall_score(all_labels, all_predictions, average=None, zero_division=0.0, labels=list(range(num_classes)))

    # Compute confusion matrix
    cm = sk_confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))

    # Compute per-class support
    class_support = np.bincount(all_labels, minlength=num_classes)

    # Build per-class metrics dict
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision_per_class_values[i]),
            "recall": float(recall_per_class_values[i]),
            "f1": float(f1_per_class_values[i]),
            "support": int(class_support[i]),
        }

    # Build F1 per class dict
    f1_per_class = {name: float(f1_per_class_values[i]) for i, name in enumerate(class_names)}

    results = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
    }

    logger.info(f"Evaluation results: Accuracy={accuracy:.4f}, F1 Macro={f1_macro:.4f}")

    return results
