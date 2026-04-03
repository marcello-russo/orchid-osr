# app/core/kfold.py
"""
K-Fold Cross-Validation con augmentation bilanciata per fold.

A differenza di cross_validate() in training.py (che lavora su dataset gia' augmentato),
questo modulo:
- Parte dalle immagini originali (non augmentate)
- Divide in fold con StratifiedKFold
- Per ogni fold: augmentation bilanciata sul train split, identity sul val split
- Nessun data leakage tra fold
"""
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import transforms as T

logger = logging.getLogger(__name__)

# Dimensioni immagini (coerenti con pipeline originale)
IMG_WIDTH = 256
IMG_HEIGHT = 512

# Augmentation functions (coerenti con pipeline originale)
AUGMENTATION_FUNCTIONS = None  # Lazy import per evitare circolarita'


def _get_augmentation_functions():
    """Lazy import delle augmentation functions."""
    global AUGMENTATION_FUNCTIONS
    if AUGMENTATION_FUNCTIONS is None:
        from app.preprocessing import rotation
        AUGMENTATION_FUNCTIONS = [
            rotation.identity,
            rotation.horizontal_flip,
            rotation.vertical_flip,
            rotation.horizontal_and_vertical_flip,
        ]
    return AUGMENTATION_FUNCTIONS


def _split_raw_images(raw_dataset_dir, class_names, num_folds, seed):
    """
    Divide le immagini originali in fold con StratifiedKFold.

    Returns:
        tuple: (all_paths, all_labels, folds) dove folds e' lista di (train_idx, val_idx)
    """
    all_paths = []
    all_labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(raw_dataset_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Classe {class_name} non trovata in {raw_dataset_dir}")
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(class_idx)

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(all_paths, all_labels))

    logger.info(f"Dataset: {len(all_paths)} immagini, {len(class_names)} classi")
    for cls_idx, cls_name in enumerate(class_names):
        count = int((all_labels == cls_idx).sum())
        logger.info(f"  {cls_name}: {count} immagini")

    return all_paths, all_labels, folds


def _copy_fold_images(all_paths, all_labels, class_names, train_idx, val_idx, fold_dir):
    """
    Copia le immagini originali nelle directory del fold.
    Anti-leakage: le immagini sono fisicamente separate prima dell'augmentation.

    Returns:
        tuple: (train_dir, val_dir)
    """
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")

    # Crea directory per classe
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Copia immagini train
    for idx in train_idx:
        class_name = class_names[all_labels[idx]]
        dst = os.path.join(train_dir, class_name, os.path.basename(all_paths[idx]))
        try:
            shutil.copy(all_paths[idx], dst)
        except OSError as e:
            logger.warning(f"Skip copia {all_paths[idx]}: {e}")

    # Copia immagini val
    for idx in val_idx:
        class_name = class_names[all_labels[idx]]
        dst = os.path.join(val_dir, class_name, os.path.basename(all_paths[idx]))
        try:
            shutil.copy(all_paths[idx], dst)
        except OSError as e:
            logger.warning(f"Skip copia {all_paths[idx]}: {e}")

    return train_dir, val_dir


def _augment_fold_train(fold_train_dir, class_names, width, height, cache_dir=None):
    """
    Augmentation bilanciata sulla directory train del fold.
    Usa augmentDataPathBalanced per classe con target calcolato sul fold corrente.

    Se cache_dir è specificato, salva l'augmented in quella directory permanente
    e la riutilizza se esiste già (l'augmentation dipende solo dal dataset, non dal modello).

    Returns:
        str: percorso della directory augmentata
    """
    from app.preprocessing.dataset_tool import augmentDataPathBalanced

    aug_functions = _get_augmentation_functions()

    # Se abbiamo una cache dir, usala come destinazione permanente
    if cache_dir and os.path.exists(cache_dir):
        total_files = sum(
            len([f for f in os.listdir(os.path.join(cache_dir, cn))
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for cn in class_names
            if os.path.exists(os.path.join(cache_dir, cn))
        )
        if total_files > 0:
            logger.info(f"  Cache augmentation trovata: {cache_dir} ({total_files} immagini)")
            return cache_dir

    augmented_dir = cache_dir if cache_dir else fold_train_dir + "_augmented"

    # Calcola il target bilanciato per questo fold
    class_sizes = {}
    for class_name in class_names:
        class_path = os.path.join(fold_train_dir, class_name)
        if os.path.exists(class_path):
            num_files = len([f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            class_sizes[class_name] = num_files
        else:
            class_sizes[class_name] = 0

    min_size = min(class_sizes.values()) if class_sizes else 0
    target_per_class = min_size * len(aug_functions)

    logger.info(f"  Augmentation target: {target_per_class} immagini/classe "
                f"(min_class={min_size}, {len(aug_functions)} augmentation)")

    # Augmenta ogni classe al target
    for class_name in class_names:
        input_path = os.path.join(fold_train_dir, class_name)
        output_path = os.path.join(augmented_dir, class_name)

        if not os.path.exists(input_path) or class_sizes.get(class_name, 0) == 0:
            continue

        os.makedirs(output_path, exist_ok=True)

        augmentDataPathBalanced(
            data_path=input_path,
            output_path=output_path,
            target_images=target_per_class,
            width=width,
            height=height,
            augmentation_functions=aug_functions,
            current_count=class_sizes[class_name],
        )

    return augmented_dir


def _prepare_fold_data(augmented_train_dir, val_dir, batch_size, seed):
    """
    Crea DataLoaders per un fold.

    Returns:
        tuple: (train_loader, val_loader)
    """
    from app.preprocessing.dataset_tool import ImageFolderWithName
    from app.preprocessing.normalization import get_mean, get_std
    from app.preprocessing.reproducibility import seed_worker
    from torch.utils.data import DataLoader

    mean = get_mean()
    std = get_std()

    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    train_dataset = ImageFolderWithName(augmented_train_dir, transform=transform)
    val_dataset = ImageFolderWithName(val_dir, transform=transform)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
    )

    logger.info(f"  Train: {len(train_dataset)} immagini ({len(train_dataset.classes)} classi)")
    logger.info(f"  Val: {len(val_dataset)} immagini ({len(val_dataset.classes)} classi)")

    return train_loader, val_loader


def _train_fold(model_variant, num_classes, train_loader, val_loader,
                num_epochs, learning_rate, device, checkpoint_path, patience=0):
    """
    Train di un singolo fold. Riutilizza train() da app.core.training.

    Returns:
        tuple: (modello, fold_metrics_dict)
    """
    from app.core.models.resnet import ResNetConfig, ResNetClassifier
    from app.core.training import train

    config = ResNetConfig(
        variant=model_variant,
        num_classes=num_classes,
        pretrained=True,
    )
    model = ResNetClassifier(config)

    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        num_classes=num_classes,
        patience=patience,
    )

    # Salva pesi migliori
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model.save_weights(checkpoint_path)

    best_f1 = max(history["val_f1"])
    best_epoch = history["val_f1"].index(best_f1) + 1
    best_acc = history["val_accuracy"][best_epoch - 1]

    fold_metrics = {
        "best_f1": float(best_f1),
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "history": {
            "train_loss": [float(x) for x in history["train_loss"]],
            "val_loss": [float(x) for x in history["val_loss"]],
            "val_accuracy": [float(x) for x in history["val_accuracy"]],
            "val_f1": [float(x) for x in history["val_f1"]],
        },
    }

    logger.info(f"  Fold completato: F1={best_f1:.4f}, Acc={best_acc:.4f}, Best epoch={best_epoch}")

    return model, fold_metrics


def _aggregate_results(fold_results):
    """
    Aggrega i risultati dei fold: mean ± std.

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


def _final_test(best_model_path, model_variant, num_classes, test_dataset_dir,
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


def _generate_report(results, variant, output_dir):
    """
    Genera un report testuale simile a REPORT_CORRETTO.txt.

    Returns:
        str: contenuto del report
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"REPORT K-FOLD CROSS VALIDATION - {variant.upper()}")
    lines.append("=" * 80)
    lines.append(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configurazione
    lines.append("CONFIGURAZIONE")
    lines.append("-" * 50)
    lines.append(f"Modello: {variant}")
    lines.append(f"Fold: {len(results['fold_results'])}")
    lines.append(f"Epochs: {results.get('num_epochs', 'N/A')}")
    lines.append(f"Batch size: {results.get('batch_size', 'N/A')}")
    lines.append(f"Learning rate: {results.get('learning_rate', 'N/A')}")
    lines.append(f"Seed: {results.get('seed', 'N/A')}")
    lines.append("")

    # Risultati aggregati
    lines.append("=" * 80)
    lines.append("RISULTATI AGGREGATI (VALIDAZIONE)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Accuracy: {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}")
    lines.append(f"F1 Macro: {results['mean_f1']:.4f} +/- {results['std_f1']:.4f}")
    lines.append(f"Miglior fold: {results['best_fold'] + 1}")
    lines.append("")

    # Breakdown per fold
    lines.append("=" * 80)
    lines.append("BREAKDOWN PER FOLD")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Fold':>6} {'Accuracy':>10} {'F1':>10} {'Val Loss':>10} {'Best Epoch':>12}")
    lines.append("-" * 50)

    for fr in results["fold_results"]:
        lines.append(
            f"{fr.get('fold', 'N/A'):>6} "
            f"{fr['best_accuracy']:>10.4f} "
            f"{fr['best_f1']:>10.4f} "
            f"{fr['final_val_loss']:>10.4f} "
            f"{fr['best_epoch']:>12}"
        )

    lines.append("")

    # Test finale
    if "test_metrics" in results and results["test_metrics"]:
        tm = results["test_metrics"]
        lines.append("=" * 80)
        lines.append("TEST FINALE SU INTERNET CROPPED")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Accuracy: {tm.get('accuracy', 0):.4f}")
        lines.append(f"F1 Macro: {tm.get('f1_macro', 0):.4f}")
        lines.append(f"Precision Macro: {tm.get('precision_macro', 0):.4f}")
        lines.append(f"Recall Macro: {tm.get('recall_macro', 0):.4f}")
        lines.append("")

        if "per_class_metrics" in tm:
            lines.append("METRICHE PER CLASSE (tutte le classi del modello):")
            lines.append("-" * 90)
            lines.append(f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
            lines.append("-" * 90)
            for cls_name, cls_metrics in tm["per_class_metrics"].items():
                note = ""
                if cls_metrics["support"] == 0:
                    note = "  (non nel test set)"
                lines.append(
                    f"{cls_name:<25} "
                    f"{cls_metrics['precision']:>10.4f} "
                    f"{cls_metrics['recall']:>10.4f} "
                    f"{cls_metrics['f1']:>10.4f} "
                    f"{cls_metrics['support']:>10} "
                    f"{cls_metrics.get('tp', 0):>6} "
                    f"{cls_metrics.get('fp', 0):>6} "
                    f"{cls_metrics.get('fn', 0):>6}"
                    f"{note}"
                )
            lines.append("")

        # Confusion matrix completa
        if "confusion_matrix" in tm and "confusion_matrix_labels" in tm:
            labels = tm["confusion_matrix_labels"]
            cm = tm["confusion_matrix"]
            lines.append("CONFUSION MATRIX (righe = true, colonne = predicted):")
            lines.append("-" * 90)
            header = f"{'':>25}"
            for lbl in labels:
                header += f" {lbl[:8]:>8}"
            lines.append(header)
            for i, row in enumerate(cm):
                row_str = f"{labels[i]:>25}"
                for val in row:
                    row_str += f" {val:>8}"
                lines.append(row_str)
            lines.append("")

    lines.append("=" * 80)
    lines.append("FINE REPORT K-FOLD")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Salva su file
    report_path = os.path.join(output_dir, f"REPORT_KFOLD_{variant}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report salvato in: {report_path}")
    return report


def kfold_experiment(
    raw_dataset_dir,
    test_dataset_dir,
    model_variant="resnet18",
    num_folds=5,
    num_epochs=26,
    batch_size=8,
    learning_rate=0.001,
    seed=57,
    num_classes=6,
    class_names=None,
    output_dir="experiments/",
    fold_index=None,
    aggregate_only=False,
    patience=0,
):
    """
    K-fold cross-validation con augmentation bilanciata per fold.

    Modalità:
    - fold_index=None, aggregate_only=False: esegue tutti i fold (default)
    - fold_index=N (1-based): esegue solo il fold N, salva risultati parziali
    - aggregate_only=True: aggrega fold completati, test finale, report

    Returns:
        dict con fold_results, mean/std metrics, test_metrics
    """
    from app.preprocessing.reproducibility import set_seed

    if class_names is None:
        class_names = [
            "O. exaltata", "O. garganica", "O. incubacea",
            "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"
        ]

    # Directory dedicata per risultati k-fold (separata dall'esperimento originale)
    kfold_dir = output_dir.rstrip('/') + f"_kfold_{model_variant}"
    os.makedirs(kfold_dir, exist_ok=True)

    # Salva configurazione esperimento
    config_path = os.path.join(kfold_dir, "config.json")
    if not os.path.exists(config_path):
        from datetime import datetime
        config = {
            "type": "kfold",
            "model_variant": model_variant,
            "num_folds": num_folds,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "num_classes": num_classes,
            "class_names": class_names,
            "raw_dataset_dir": raw_dataset_dir,
            "test_dataset_dir": test_dataset_dir,
            "patience": patience,
            "timestamp": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config salvato in: {config_path}")

    # --- MODALITA: AGGREGAZIONE ---
    if aggregate_only:
        return _aggregate_and_test(
            output_dir=kfold_dir,
            model_variant=model_variant,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            test_dataset_dir=test_dataset_dir,
            num_classes=num_classes,
            class_names=class_names,
        )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info(f"K-FOLD CROSS VALIDATION - {model_variant.upper()}")
    logger.info(f"Dataset: {raw_dataset_dir}")
    logger.info(f"Test: {test_dataset_dir}")
    logger.info(f"Folds: {num_folds}, Epochs: {num_epochs}, Batch size: {batch_size}, Patience: {patience}")
    if fold_index is not None:
        logger.info(f"MODALITA: fold singolo {fold_index}/{num_folds}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)

    # Step 1: Split immagini originali
    all_paths, all_labels, folds = _split_raw_images(
        raw_dataset_dir, class_names, num_folds, seed
    )

    # Se fold_index specificato, esegui solo quel fold
    if fold_index is not None:
        fold_indices = [fold_index - 1]  # Converti 1-based → 0-based
    else:
        fold_indices = list(range(num_folds))

    base_temp_dir = os.path.join(os.path.dirname(raw_dataset_dir), "temp_kfold")
    # Cache permanente per augmentation (dipende solo dal dataset, non dal modello)
    base_aug_cache = os.path.join(os.path.dirname(raw_dataset_dir), "kfold_augmented")

    try:
        for fold_idx in fold_indices:
            train_idx, val_idx = folds[fold_idx]

            logger.info(f"\n{'=' * 50}")
            logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
            logger.info(f"Train: {len(train_idx)} immagini, Val: {len(val_idx)} immagini")
            logger.info(f"{'=' * 50}")

            fold_dir = os.path.join(base_temp_dir, f"fold_{fold_idx}")
            # Cache augmentation permanente per questo fold
            aug_cache_dir = os.path.join(base_aug_cache, f"fold_{fold_idx}")

            try:
                # Step 2: Copia immagini in directory del fold
                train_dir, val_dir = _copy_fold_images(
                    all_paths, all_labels, class_names,
                    train_idx, val_idx, fold_dir
                )

                # Step 3: Augmentation bilanciata sul train (con cache)
                augmented_dir = _augment_fold_train(
                    train_dir, class_names, IMG_WIDTH, IMG_HEIGHT,
                    cache_dir=aug_cache_dir
                )

                # Step 4: Crea DataLoaders
                train_loader, val_loader = _prepare_fold_data(
                    augmented_dir, val_dir, batch_size, seed
                )

                # Step 5: Train
                checkpoint_path = os.path.join(kfold_dir, f"best_fold_{fold_idx}.pt")
                model, fold_metrics = _train_fold(
                    model_variant, num_classes, train_loader, val_loader,
                    num_epochs, learning_rate, device, checkpoint_path,
                    patience=patience,
                )

                fold_metrics["fold"] = fold_idx + 1
                fold_metrics["batch_size"] = batch_size
                fold_metrics["train_size"] = len(train_idx)
                fold_metrics["val_size"] = len(val_idx)

                # Salva risultati fold singolo
                fold_json_path = os.path.join(kfold_dir, f"fold_{fold_idx}.json")
                with open(fold_json_path, "w") as f:
                    json.dump(fold_metrics, f, indent=2, default=str)
                logger.info(f"Fold {fold_idx + 1} salvato in: {fold_json_path}")

            finally:
                # Cleanup directory temporanea raw del fold (NON la cache augmented)
                if os.path.exists(fold_dir):
                    shutil.rmtree(fold_dir, ignore_errors=True)

    finally:
        # Cleanup globale
        if os.path.exists(base_temp_dir):
            shutil.rmtree(base_temp_dir, ignore_errors=True)

    # Se singolo fold, restituisci risultati parziali
    if fold_index is not None:
        logger.info(f"\nFold {fold_index} completato.")
        logger.info(f"Per aggregare tutti i fold: --cross-validate --exp 4 --model {model_variant} --aggregate-only")
        return {"fold_completed": fold_index, "fold_metrics": fold_metrics}

    # --- Tutti i fold completati: aggrega e testa ---
    return _aggregate_and_test(
        output_dir=kfold_dir,
        model_variant=model_variant,
        num_folds=num_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        test_dataset_dir=test_dataset_dir,
        num_classes=num_classes,
        class_names=class_names,
    )


def _aggregate_and_test(
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
    aggregated = _aggregate_results(fold_results)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"RISULTATI K-FOLD ({len(fold_results)} folds)")
    logger.info(f"{'=' * 50}")
    logger.info(f"Accuracy: {aggregated['mean_accuracy']:.4f} +/- {aggregated['std_accuracy']:.4f}")
    logger.info(f"F1 Macro: {aggregated['mean_f1']:.4f} +/- {aggregated['std_f1']:.4f}")

    # Test finale con modello migliore
    best_fold = aggregated["best_fold"]
    best_model_path = os.path.join(output_dir, f"best_fold_{best_fold}.pt")

    logger.info(f"\nTest finale con modello fold {best_fold + 1} su {test_dataset_dir}")
    test_metrics = _final_test(
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
    _generate_report(results, model_variant, output_dir)

    return results
