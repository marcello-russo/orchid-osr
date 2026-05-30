# app/cli/reports.py
"""
Report generation for experiment results.
"""

from datetime import datetime


def _generate_experiment_report(results, model_variant, exp, epochs, batch_size, learning_rate, output_path):
    """Genera un report testuale per l'esperimento standard."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"REPORT ESPERIMENTO - {exp['name'].upper()} - {model_variant.upper()}")
    lines.append("=" * 80)
    lines.append(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configurazione
    lines.append("CONFIGURAZIONE")
    lines.append("-" * 50)
    lines.append(f"Esperimento: {exp['name']}")
    lines.append(f"Descrizione: {exp['description']}")
    lines.append(f"Modello: {model_variant}")
    lines.append(f"Train dataset: {exp['train_dataset']}")
    lines.append(f"Test dataset: {exp['test_dataset']}")
    lines.append(f"Epochs: {epochs}")
    lines.append(f"Batch size: {batch_size}")
    lines.append(f"Learning rate: {learning_rate}")
    lines.append(f"Classi test: {exp['test_classes']}")
    lines.append("")

    # Metriche aggregate
    lines.append("=" * 80)
    lines.append("RISULTATI TEST")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Accuracy: {results['accuracy']:.4f}")
    lines.append(f"F1 Macro: {results['f1_macro']:.4f}")
    lines.append(f"Precision Macro: {results.get('precision_macro', 0):.4f}")
    lines.append(f"Recall Macro: {results.get('recall_macro', 0):.4f}")
    lines.append("")

    # Metriche per classe
    if "per_class_metrics" in results:
        lines.append("METRICHE PER CLASSE:")
        lines.append("-" * 70)
        lines.append(f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 70)
        for cls_name, cls_metrics in results["per_class_metrics"].items():
            lines.append(
                f"{cls_name:<25} "
                f"{cls_metrics['precision']:>10.4f} "
                f"{cls_metrics['recall']:>10.4f} "
                f"{cls_metrics['f1']:>10.4f} "
                f"{cls_metrics['support']:>10}"
            )
        lines.append("")

    # Confusion matrix
    if "confusion_matrix" in results and "per_class_metrics" in results:
        class_names = list(results["per_class_metrics"].keys())
        cm = results["confusion_matrix"]
        lines.append("CONFUSION MATRIX (righe = true, colonne = predicted):")
        lines.append("-" * 70)
        header = f"{'':>25}"
        for lbl in class_names:
            header += f" {lbl[:8]:>8}"
        lines.append(header)
        for i, row in enumerate(cm):
            row_str = f"{class_names[i]:>25}"
            for val in row:
                row_str += f" {val:>8}"
            lines.append(row_str)
        lines.append("")

    lines.append("=" * 80)
    lines.append("FINE REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
