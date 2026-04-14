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

        # Best epoch marker (best_epoch is 1-based, x-axis is 1-based)
        best_ep = fold["best_epoch"]
        if best_ep <= len(epochs):
            axes[0].axvline(x=best_ep, color=color, alpha=0.3, linestyle=":")
            axes[1].axvline(x=best_ep, color=color, alpha=0.3, linestyle=":")

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


def plot_fold_summary(data, output_dir):
    """Horizontal bar chart: F1 and Accuracy per fold with mean +/- std."""
    folds = data["folds"]
    results = data.get("results")

    fold_labels = [f"Fold {f.get('fold', i + 1)}" for i, f in enumerate(folds)]
    f1_values = [f["best_f1"] for f in folds]
    acc_values = [f["best_accuracy"] for f in folds]

    y = np.arange(len(folds))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.barh(y + height / 2, f1_values, height, label="F1", color=COLORS[0])
    bars2 = ax.barh(y - height / 2, acc_values, height, label="Accuracy", color=COLORS[2])

    # Annotate values on bars
    for bar in bars1:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=9)
    for bar in bars2:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=9)

    # Mean +/- std annotation
    if results and "mean_f1" in results:
        mean_f1 = results["mean_f1"]
        std_f1 = results.get("std_f1", 0)
        ax.set_title(f"Fold Summary — Mean F1: {mean_f1:.3f} +/- {std_f1:.3f}")
    else:
        ax.set_title("Fold Summary")

    ax.set_xlabel("Score")
    ax.set_yticks(y)
    ax.set_yticklabels(fold_labels)
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(max(f1_values), max(acc_values)) + 0.1)

    fig.tight_layout()
    path = os.path.join(output_dir, "fold_summary.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Salvato: {path}")


def plot_confusion_matrix(data, output_dir):
    """Heatmap of test confusion matrix with annotations."""
    results = data.get("results")
    if not results or "test_metrics" not in results:
        print("Nessuna confusion matrix disponibile, skip")
        return

    cm = np.array(results["test_metrics"]["confusion_matrix"])
    labels = results["test_metrics"].get("confusion_matrix_labels",
                                          [str(i) for i in range(len(cm))])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    # Annotate each cell
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=11)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Salvato: {path}")


def plot_overfitting_proxy(data, output_dir):
    """
    Ratio val_loss / train_loss per epoch per fold.
    Values > 1 indicate potential overfitting.
    """
    folds = data["folds"]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, fold in enumerate(folds):
        color = COLORS[i % len(COLORS)]
        train = np.array(fold["history"]["train_loss"])
        val = np.array(fold["history"]["val_loss"])
        # Avoid division by zero
        train_safe = np.where(train == 0, 1e-10, train)
        ratio = val / train_safe
        epochs = range(1, len(ratio) + 1)
        label = f"Fold {fold.get('fold', i + 1)}"
        ax.plot(epochs, ratio, color=color, label=label)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7,
               label="Overfitting threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss / train_loss")
    ax.set_title("Overfitting Proxy (val_loss / train_loss)")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(output_dir, "overfitting_proxy.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Salvato: {path}")
