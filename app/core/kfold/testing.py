# app/core/kfold/testing.py
"""Testing and aggregation functions for k-fold cross-validation."""
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

logger = logging.getLogger(__name__)

# Dimensioni immagini (coerenti con pipeline originale)
IMG_WIDTH = 256
IMG_HEIGHT = 512


def aggregate_results(fold_results):
    """
    Aggrega i risultati dei fold: mean +- std.

    Returns:
        dict con mean_accuracy, std_accuracy, mean_f1, std_f1, best_fold
    """
    all_f1 = [r["best_f1"] for r in fold_results]
    all_acc = [r["best_accuracy"] for r in fold_results]

    best_fold_idx = int(np.argmax(all_f1))

    return {
        "mean_accuracy": float(np.mean(all_acc)),
        "std_accuracy": float(np.std(all_acc)),
        "mean_f1": float(np.mean(all_f1)),
        "std_f1": float(np.std(all_f1)),
        "best_fold": best_fold_idx,
        "fold_results": fold_results,
    }


def final_test(best_model_path, model_variant, num_classes, test_dataset_dir,
               class_names, batch_size, device):
    """
    Valutazione finale del modello migliore sul test set.

    Gestisce il mapping delle classi: il modello ha 6 classi (class_names),
    il test set puo' avere un sottoinsieme con indici diversi (es. ImageFolder
    assegna O. sphegodes all'indice 3, ma nel modello e' all'indice 4).

    Returns:
        dict con metriche complete
    """
    from app.core.models.resnet import ResNetConfig, ResNetClassifier
    from app.preprocessing.dataset_tool import ImageFolderWithName
    from app.preprocessing.normalization import get_mean, get_std
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix as sk_confusion_matrix,
    )

    mean = get_mean()
    std = get_std()

    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # Carica modello
    config = ResNetConfig(variant=model_variant, num_classes=num_classes, pretrained=True)
    model = ResNetClassifier(config)
    model.load_weights(best_model_path)
    model = model.to(device)

    # Carica test set
    test_dataset = ImageFolderWithName(test_dataset_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_class_names = test_dataset.classes
    logger.info(f"Test set: {len(test_dataset)} immagini, classi: {test_class_names}")
    logger.info(f"Modello: {num_classes} classi, classi modello: {class_names}")

    # Mapping: test_class_names -> model class indices
    model_class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    test_to_model = {}
    for test_idx, cls_name in enumerate(test_class_names):
        if cls_name in model_class_to_idx:
            test_to_model[test_idx] = model_class_to_idx[cls_name]
        else:
            logger.warning(f"Classe '{cls_name}' del test set non trovata nel modello!")

    logger.info(f"Mapping test->modello: {test_to_model}")

    # Raccolta predizioni
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels_raw = np.array(all_labels)

    # Remappa le label del test set sugli indici del modello
    all_labels_mapped = np.array([test_to_model[l] for l in all_labels_raw])

    # Classi presenti nel test set (indici nel modello)
    present_indices = sorted(test_to_model.values())

    # Calcola metriche aggregate solo sulle classi presenti nel test
    accuracy = accuracy_score(all_labels_mapped, all_predictions)
    f1_macro = f1_score(all_labels_mapped, all_predictions, average="macro",
                        zero_division=0.0, labels=present_indices)
    precision_macro = precision_score(all_labels_mapped, all_predictions, average="macro",
                                      zero_division=0.0, labels=present_indices)
    recall_macro = recall_score(all_labels_mapped, all_predictions, average="macro",
                                zero_division=0.0, labels=present_indices)

    # Confusion matrix completa: righe = tutte le classi del modello, colonne = tutte le classi del modello
    all_model_indices = list(range(num_classes))
    cm_full = sk_confusion_matrix(all_labels_mapped, all_predictions, labels=all_model_indices)

    # Metriche per TUTTE le classi del modello
    per_class_metrics = {}
    f1_per_class = {}

    for model_idx in all_model_indices:
        cls_name = class_names[model_idx]
        support = int((all_labels_mapped == model_idx).sum())
        tp = int(cm_full[model_idx, model_idx])
        fp = int(cm_full[:, model_idx].sum()) - tp
        fn = int(cm_full[model_idx, :].sum()) - tp

        if support > 0:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            # Classe non presente nel test set: solo falsi positivi
            prec = 0.0
            rec = 0.0
            f1 = 0.0

        per_class_metrics[cls_name] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_per_class[cls_name] = float(f1)

    results = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm_full.tolist(),
        "confusion_matrix_labels": class_names,
        "per_class_metrics": per_class_metrics,
    }

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Macro: {f1_macro:.4f} (solo classi presenti nel test)")
    for model_idx in all_model_indices:
        cls_name = class_names[model_idx]
        m = per_class_metrics[cls_name]
        status = f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
        if m["support"] == 0:
            status += f"  (non nel test, {m['fp']} falsi positivi)"
        logger.info(f"  {cls_name}: {status}")

    return results


def aggregate_and_test(
    output_dir,
    model_variant,
    num_folds,
    num_epochs,
    batch_size,
    learning_rate,
    seed,
    test_dataset_dir,
    num_classes,
    class_names,
):
    """
    Legge i risultati dei fold completati, aggrega, fa test finale e genera report.
    """
    import json
    from app.core.kfold.reporting import generate_report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Leggi risultati fold completati
    fold_results = []
    for fold_idx in range(num_folds):
        fold_json = os.path.join(output_dir, f"fold_{fold_idx}.json")
        if os.path.exists(fold_json):
            with open(fold_json, "r") as f:
                fold_results.append(json.load(f))
        else:
            logger.warning(f"Fold {fold_idx + 1} non trovato: {fold_json}")

    if not fold_results:
        logger.error("Nessun fold completato trovato. Esegui prima i fold singoli.")
        return None

    logger.info(f"Trovati {len(fold_results)}/{num_folds} fold completati")

    # Leggi batch_size effettivo dai fold JSON (piu' affidabile del parametro)
    effective_batch_size = batch_size
    if fold_results and "batch_size" in fold_results[0]:
        effective_batch_size = fold_results[0]["batch_size"]
        if effective_batch_size != batch_size:
            logger.warning(
                f"batch_size dal parametro ({batch_size}) != batch_size dai fold ({effective_batch_size}). "
                f"Uso il valore dai fold."
            )

    # Aggrega
    aggregated = aggregate_results(fold_results)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"RISULTATI K-FOLD ({len(fold_results)} folds)")
    logger.info(f"{'=' * 50}")
    logger.info(f"Accuracy: {aggregated['mean_accuracy']:.4f} +/- {aggregated['std_accuracy']:.4f}")
    logger.info(f"F1 Macro: {aggregated['mean_f1']:.4f} +/- {aggregated['std_f1']:.4f}")

    # Test finale con modello migliore
    best_fold = aggregated["best_fold"]
    best_model_path = os.path.join(output_dir, f"best_fold_{best_fold}.pt")

    logger.info(f"\nTest finale con modello fold {best_fold + 1} su {test_dataset_dir}")
    test_metrics = final_test(
        best_model_path, model_variant, num_classes, test_dataset_dir,
        class_names, batch_size, device
    )

    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Macro: {test_metrics['f1_macro']:.4f}")

    # Risultato completo
    results = {
        **aggregated,
        "model_variant": model_variant,
        "num_folds": num_folds,
        "num_epochs": num_epochs,
        "batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_model_path": best_model_path,
        "test_metrics": test_metrics,
    }

    # Salva JSON
    json_path = os.path.join(output_dir, f"results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Risultati salvati in: {json_path}")

    # Genera report
    generate_report(results, model_variant, output_dir)

    return results
