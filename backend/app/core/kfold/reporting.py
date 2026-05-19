# app/core/kfold/reporting.py
"""Report generation for k-fold cross-validation."""
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_report(results, variant, output_dir):
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
    lines.append(f"Miglior fold: {results['best_fold']}")
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
        lines.append("TEST FINALE")
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
            lines.append(
                f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}"
                f" {'Support':>10} {'TP':>6} {'FP':>6} {'FN':>6}"
            )
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
