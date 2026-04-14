"""Plotting functions for k-fold experiment results."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.visualization.style import COLORS, FIGURE_SIZE, DPI


def plot_loss_curves(data, output_dir):
    """
    Plot train loss and val loss per epoch, one line per fold.
    Vertical dashed line at each fold's best_epoch.
    """
    folds = data["folds"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, fold in enumerate(folds):
        color = COLORS[i % len(COLORS)]
        epochs = range(1, len(fold["history"]["train_loss"]) + 1)
        label = f"Fold {fold.get('fold', i + 1)}"

        axes[0].plot(epochs, fold["history"]["train_loss"], color=color,
                     label=label, linestyle="--")
        axes[1].plot(epochs, fold["history"]["val_loss"], color=color,
                     label=label)

        # Best epoch marker
        best_ep = fold["best_epoch"]
        if best_ep < len(epochs):
            axes[0].axvline(x=best_ep + 1, color=color, alpha=0.3, linestyle=":")
            axes[1].axvline(x=best_ep + 1, color=color, alpha=0.3, linestyle=":")

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    axes[1].legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(output_dir, "loss_curves.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Salvato: {path}")


def plot_metric_curves(data, output_dir):
    """
    Plot val F1 and val accuracy per epoch, one line per fold.
    Horizontal dashed line at mean best F1.
    """
    folds = data["folds"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mean_best_f1 = np.mean([f["best_f1"] for f in folds])

    for i, fold in enumerate(folds):
        color = COLORS[i % len(COLORS)]
        epochs = range(1, len(fold["history"]["val_f1"]) + 1)
        label = f"Fold {fold.get('fold', i + 1)}"

        axes[0].plot(epochs, fold["history"]["val_f1"], color=color,
                     label=label, marker="o", markevery=5, markersize=4)
        axes[1].plot(epochs, fold["history"]["val_accuracy"], color=color,
                     label=label, marker="o", markevery=5, markersize=4)

    axes[0].axhline(y=mean_best_f1, color="gray", linestyle="--",
                    alpha=0.7, label=f"Mean best F1: {mean_best_f1:.3f}")
    axes[0].set_title("Validation F1 Score")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("F1 Score")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    axes[0].legend(loc="lower right", framealpha=0.9)
    axes[1].legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(output_dir, "metric_curves.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Salvato: {path}")
