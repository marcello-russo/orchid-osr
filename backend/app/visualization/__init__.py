"""Visualization module: generate plots from k-fold experiment results."""

import os

from app.visualization.plots import (
    plot_confusion_matrix,
    plot_fold_summary,
    plot_loss_curves,
    plot_metric_curves,
    plot_overfitting_proxy,
)
from app.visualization.style import setup_style
from app.visualization.utils import load_experiment_data


def plot_experiment(exp_dir, output_dir=None):
    """Generate all plots for a k-fold experiment.

    Args:
    ----
        exp_dir: path to experiment directory (contains fold_*.json, results.json)
        output_dir: where to save PNGs (default: exp_dir/figures/)

    """
    if output_dir is None:
        output_dir = os.path.join(exp_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    data = load_experiment_data(exp_dir)
    setup_style()

    print(f"Generazione grafici per: {exp_dir}")
    print(f"Modello: {data['config'].get('model_variant', 'unknown')}")
    print(f"Folds: {len(data['folds'])}")

    plot_loss_curves(data, output_dir)
    plot_metric_curves(data, output_dir)
    plot_fold_summary(data, output_dir)
    plot_confusion_matrix(data, output_dir)
    plot_overfitting_proxy(data, output_dir)

    print(f"\nTutti i grafici salvati in: {output_dir}")
