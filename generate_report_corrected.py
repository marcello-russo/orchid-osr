#!/usr/bin/env python3
"""
Genera un report corretto per gli esperimenti di classificazione.
Corregge il problema di mapping delle classi per Exp4 dove le label del test set
non corrispondono agli indici di output del modello.
"""

import numpy as np
from pathlib import Path

# Class definitions
ALL_CLASSES = [
    "O. exaltata",      # 0
    "O. garganica",     # 1
    "O. incubacea",     # 2
    "O. majellensis",   # 3
    "O. sphegodes",     # 4
    "O. sphegodes_Palena"  # 5
]

# When internet test set is loaded with ImageFolder, classes are assigned alphabetically:
INTERNET_CLASSES = [
    "O. exaltata",      # 0 in test set
    "O. garganica",     # 1 in test set
    "O. incubacea",     # 2 in test set
    "O. sphegodes"      # 3 in test set (but 4 in model!)
]

# Mapping from test set indices to model indices for Exp4
# Test index -> Model index
INTERNET_TO_ALL_MAPPING = {
    0: 0,  # O. exaltata
    1: 1,  # O. garganica
    2: 2,  # O. incubacea
    3: 4,  # O. sphegodes (test index 3, model index 4)
}

# Confusion matrices from logs (RUN: 2026-03-10 with original config: batch=8, lr=0.001, no class weights)
# For Exp1, the matrix is already correct (model indices = test indices)
# For Exp2-4, we need to remap the rows (true labels) from test indices to model indices
CONFUSION_MATRICES_RAW = {
    "Exp1": np.array([
        [54,  0,  0,  0,  2,  0],
        [ 1, 45,  4,  0,  0,  0],
        [ 0,  3, 28,  0,  0,  0],
        [ 0,  0,  0, 39,  1,  0],
        [ 1,  0,  1,  0, 18,  0],
        [ 0,  0,  0,  0,  0, 52]
    ], dtype=np.int32),

    # Exp2-4: rows are test set indices (0-3), columns are model predictions (0-5)
    "Exp2": np.array([
        [161,  10,  10,  14,   2,  21],
        [ 84,  95,  45,  65,   1,  29],
        [ 36,  27, 118,  18,   0,  11],
        [135,  18,  38,  81,   5,  56]
    ], dtype=np.int32),

    "Exp3": np.array([
        [131,  28,   4,  16,   0,  39],
        [  1, 282,   6,  16,   0,  14],
        [  5,  96,  90,   3,   1,  15],
        [ 39,  73,  18, 112,  17,  74]
    ], dtype=np.int32),

    "Exp4": np.array([
        [206,   1,   0,   0,  10,   1],
        [  2, 306,   4,   0,   7,   0],
        [  1,   4, 203,   0,   2,   0],
        [  5,   9,   5,   0, 313,   1]
    ], dtype=np.int32),

    # Exp5: Combined -> Privato (6 classi, nessun mapping necessario)
    "Exp5": np.array([
        [55,  0,  0,  0,  1,  0],
        [ 0, 48,  1,  0,  0,  1],
        [ 0,  2, 29,  0,  0,  0],
        [ 0,  0,  0, 40,  0,  0],
        [ 0,  0,  1,  1, 17,  1],
        [ 0,  0,  0,  0,  0, 52]
    ], dtype=np.int32)
}


def calculate_metrics_with_mapping(cm_raw, test_to_model_mapping, model_classes_count):
    """
    Calculate metrics from confusion matrix with proper class mapping.

    Args:
        cm_raw: Raw confusion matrix (test_indices x model_indices)
        test_to_model_mapping: Dict mapping test index -> model index
        model_classes_count: Number of classes the model outputs
    """
    per_class_metrics = {}
    total_samples = 0
    correct = 0

    for test_idx, model_idx in test_to_model_mapping.items():
        # True positives: prediction matches the correct model class
        tp = int(cm_raw[test_idx, model_idx])

        # Support: total samples of this class in test set
        support = int(cm_raw[test_idx, :].sum())

        # False negatives: samples of this class predicted as something else
        fn = support - tp

        # False positives: samples of other classes predicted as this class
        fp = int(cm_raw[:, model_idx].sum()) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[model_idx] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

        total_samples += support
        correct += tp

    accuracy = correct / total_samples if total_samples > 0 else 0

    # Macro averages
    macro_precision = np.mean([m["precision"] for m in per_class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "total_samples": total_samples,
        "correct": correct
    }


def calculate_metrics_exp1(cm):
    """Calculate metrics for Exp1 (6 classes, no mapping needed)."""
    per_class_metrics = {}
    total = int(cm.sum())
    correct = 0

    for i in range(6):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[i] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "support": support
        }
        correct += tp

    return {
        "accuracy": correct / total,
        "macro_precision": np.mean([m["precision"] for m in per_class_metrics.values()]),
        "macro_recall": np.mean([m["recall"] for m in per_class_metrics.values()]),
        "macro_f1": np.mean([m["f1"] for m in per_class_metrics.values()]),
        "per_class": per_class_metrics,
        "total_samples": total,
        "correct": correct
    }


def generate_report():
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("REPORT COMPLETO CORRETTO - ESPERIMENTI CLASSIFICAZIONE ORCHIDEE")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append("NOTA IMPORTANTE: Questo report corregge il problema di mapping delle classi")
    report_lines.append("per Exp4. Il test set internet assegna O. sphegodes all'indice 3, mentre")
    report_lines.append("il modello combined si aspetta O. sphegodes all'indice 4.")
    report_lines.append("")

    # Experiment configurations
    report_lines.append("CONFIGURAZIONE ESPERIMENTI")
    report_lines.append("-" * 50)
    report_lines.append("Exp1: Train Privato -> Test Privato (baseline, 6 classi)")
    report_lines.append("Exp2: Train Privato -> Test Internet Raw (4 classi)")
    report_lines.append("Exp3: Train Privato -> Test Internet Cropped (4 classi)")
    report_lines.append("Exp4: Train Combined -> Test Internet Cropped (4 classi, mapping corretto)")
    report_lines.append("Exp5: Train Combined -> Test Privato (6 classi, verifica catastrophic forgetting)")
    report_lines.append("")

    # Calculate metrics
    all_metrics = {}

    # Exp1 e Exp5: 6 classi, nessun mapping necessario
    all_metrics["Exp1"] = calculate_metrics_exp1(CONFUSION_MATRICES_RAW["Exp1"])
    all_metrics["Exp5"] = calculate_metrics_exp1(CONFUSION_MATRICES_RAW["Exp5"])

    # Exp2-4 with proper mapping
    for exp in ["Exp2", "Exp3", "Exp4"]:
        all_metrics[exp] = calculate_metrics_with_mapping(
            CONFUSION_MATRICES_RAW[exp],
            INTERNET_TO_ALL_MAPPING,
            6
        )

    # Summary table
    report_lines.append("=" * 90)
    report_lines.append("RISULTATI COMPLESSIVI (CORRETTI)")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append(f"{'Esperimento':<40} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10}")
    report_lines.append("-" * 80)

    exp_labels = {
        "Exp1": "Exp1: Privato -> Privato",
        "Exp2": "Exp2: Privato -> Internet Raw",
        "Exp3": "Exp3: Privato -> Internet Cropped",
        "Exp4": "Exp4: Combined -> Internet Cropped",
        "Exp5": "Exp5: Combined -> Privato"
    }

    for exp in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]:
        m = all_metrics[exp]
        report_lines.append(
            f"{exp_labels[exp]:<40} {m['accuracy']*100:>9.2f}% {m['macro_f1']*100:>9.2f}% "
            f"{m['macro_precision']*100:>9.2f}% {m['macro_recall']*100:>9.2f}%"
        )

    report_lines.append("")

    # Per-class breakdown
    for exp in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]:
        m = all_metrics[exp]

        report_lines.append("=" * 90)
        report_lines.append(f"ANALISI DETTAGLIATA: {exp_labels[exp]}")
        report_lines.append("=" * 90)
        report_lines.append("")
        report_lines.append(f"Accuracy: {m['accuracy']*100:.2f}% ({m['correct']}/{m['total_samples']})")
        report_lines.append(f"F1 Macro: {m['macro_f1']*100:.2f}%")
        report_lines.append(f"Precision Macro: {m['macro_precision']*100:.2f}%")
        report_lines.append(f"Recall Macro: {m['macro_recall']*100:.2f}%")
        report_lines.append("")
        report_lines.append("METRICHE PER CLASSE:")
        report_lines.append("-" * 90)
        report_lines.append(f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
        report_lines.append("-" * 90)

        for cls_idx in sorted(m["per_class"].keys()):
            cls_metrics = m["per_class"][cls_idx]
            cls_name = ALL_CLASSES[cls_idx]
            report_lines.append(
                f"{cls_name:<25} {cls_metrics['precision']*100:>9.2f}% {cls_metrics['recall']*100:>9.2f}% "
                f"{cls_metrics['f1']*100:>9.2f}% {cls_metrics['support']:>10} {cls_metrics['tp']:>6} "
                f"{cls_metrics['fp']:>6} {cls_metrics['fn']:>6}"
            )

        report_lines.append("")

    # Analysis
    report_lines.append("=" * 90)
    report_lines.append("ANALISI E CONCLUSIONI")
    report_lines.append("=" * 90)
    report_lines.append("")

    report_lines.append("1. BASELINE (Exp1):")
    report_lines.append(f"   - F1 Score: {all_metrics['Exp1']['macro_f1']*100:.2f}%")
    report_lines.append("   - Il modello privato performa eccellentemente sul dataset privato")
    report_lines.append("")

    report_lines.append("2. DOMAIN SHIFT (Exp2 vs Exp3):")
    report_lines.append(f"   - Exp2 (internet raw): F1 = {all_metrics['Exp2']['macro_f1']*100:.2f}%")
    report_lines.append(f"   - Exp3 (internet cropped): F1 = {all_metrics['Exp3']['macro_f1']*100:.2f}%")
    report_lines.append("   - Il cropping migliora significativamente le performance (+{:.1f}%)".format(
        (all_metrics['Exp3']['macro_f1'] - all_metrics['Exp2']['macro_f1'])*100))
    report_lines.append("")

    report_lines.append("3. DATA AUGMENTATION (Exp4):")
    report_lines.append(f"   - Exp4 (combined): F1 = {all_metrics['Exp4']['macro_f1']*100:.2f}%")
    report_lines.append(f"   - Accuracy: {all_metrics['Exp4']['accuracy']*100:.2f}%")
    report_lines.append("   - Questo e' il MIGLIOR risultato sui dati internet!")
    report_lines.append("   - Miglioramento rispetto a Exp3: +{:.1f}% F1".format(
        (all_metrics['Exp4']['macro_f1'] - all_metrics['Exp3']['macro_f1'])*100))
    report_lines.append("")
    report_lines.append("   CORREZIONE APPLICATA:")
    report_lines.append("   - Il modello combined predice correttamente O. sphegodes (classe 4)")
    report_lines.append("   - Il test set aveva label 3 per O. sphegodes (errato)")
    report_lines.append("   - Dopo la correzione: O. sphegodes ha F1 = {:.2f}%".format(
        all_metrics['Exp4']['per_class'][4]['f1']*100))
    report_lines.append("")

    report_lines.append("4. CATASTROPHIC FORGETTING (Exp5):")
    report_lines.append(f"   - Exp5 (combined -> privato): F1 = {all_metrics['Exp5']['macro_f1']*100:.2f}%")
    report_lines.append(f"   - Accuracy: {all_metrics['Exp5']['accuracy']*100:.2f}%")
    report_lines.append("   - Il modello combined mantiene ottime performance sul dataset privato")
    report_lines.append("   - NESSUN catastrophic forgetting rilevato!")
    report_lines.append("   - Confronto Exp1 vs Exp5: {:.2f}% vs {:.2f}% F1".format(
        all_metrics['Exp1']['macro_f1']*100, all_metrics['Exp5']['macro_f1']*100))
    report_lines.append("")

    # Per-class comparison
    report_lines.append("5. CONFRONTO PER-CLASSE (F1 Score):")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Classe':<20} {'Exp1':>10} {'Exp2':>10} {'Exp3':>10} {'Exp4':>10} {'Exp5':>10}")
    report_lines.append("-" * 80)

    internet_class_indices = [0, 1, 2, 4]  # O. exaltata, O. garganica, O. incubacea, O. sphegodes
    for cls_idx in internet_class_indices:
        cls_name = ALL_CLASSES[cls_idx]
        row = f"{cls_name:<20}"
        for exp in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]:
            if cls_idx in all_metrics[exp]["per_class"]:
                f1 = all_metrics[exp]["per_class"][cls_idx]["f1"] * 100
                row += f" {f1:>9.2f}%"
            else:
                row += f" {'N/A':>10}"
        report_lines.append(row)

    report_lines.append("")

    # Key findings
    report_lines.append("=" * 90)
    report_lines.append("FINDINGS CHIAVE")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append("[OK] Il modello privato performa bene sul proprio test set (F1={:.2f}%)".format(all_metrics['Exp1']['macro_f1']*100))
    report_lines.append("[OK] Il domain shift riduce significativamente le performance")
    report_lines.append("[OK] Il cropping delle immagini migliora le performance del +{:.1f}%".format(
        (all_metrics['Exp3']['macro_f1'] - all_metrics['Exp2']['macro_f1'])*100))
    report_lines.append("[OK] L'addestramento su dataset combined (privato + internet) da' i MIGLIORI risultati")
    report_lines.append("     su dati internet (F1={:.2f}%), superando di molto il modello solo privato".format(all_metrics['Exp4']['macro_f1']*100))
    report_lines.append("[OK] NESSUN catastrophic forgetting: il modello combined mantiene performance")
    report_lines.append("     eccellenti sul dataset privato originale (Exp5: F1={:.2f}%)".format(all_metrics['Exp5']['macro_f1']*100))
    report_lines.append("")
    report_lines.append("[OK] Exp4 mostra che l'aggiunta di dati internet al training set e' molto efficace:")
    report_lines.append("     - O. exaltata: {:.2f}% F1".format(all_metrics['Exp4']['per_class'][4]['f1']*100 if 4 in all_metrics['Exp4']['per_class'] else 0))
    report_lines.append("     - O. garganica: {:.2f}% F1".format(all_metrics['Exp4']['per_class'][1]['f1']*100 if 1 in all_metrics['Exp4']['per_class'] else 0))
    report_lines.append("     - O. incubacea: {:.2f}% F1".format(all_metrics['Exp4']['per_class'][2]['f1']*100 if 2 in all_metrics['Exp4']['per_class'] else 0))
    report_lines.append("")

    # Training info
    report_lines.append("=" * 90)
    report_lines.append("INFORMAZIONI TRAINING")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append("Configurazione (stessa dell'originale Di Pierro):")
    report_lines.append("  - ResNet18 pretrained su ImageNet")
    report_lines.append("  - 26 epochs")
    report_lines.append("  - batch_size = 8")
    report_lines.append("  - learning_rate = 0.001")
    report_lines.append("  - Class weights: No")
    report_lines.append("  - Augmentation: bilanciata a 280 immagini/classe")
    report_lines.append("")
    report_lines.append("Modello Privato (Exp1-3):")
    report_lines.append("  - Train: dataset privato (1252 immagini, 6 classi)")
    report_lines.append("")
    report_lines.append("Modello Combined (Exp4-5):")
    report_lines.append("  - Train: privato (1252) + internet cropped (1080) = 2332 immagini")
    report_lines.append("")

    report_lines.append("=" * 90)
    report_lines.append("FINE REPORT CORRETTO")
    report_lines.append("=" * 90)

    return "\n".join(report_lines)


if __name__ == "__main__":
    report = generate_report()
    print(report)

    output_path = Path(__file__).parent / "results" / "REPORT_CORRETTO.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport salvato in: {output_path}")
