"""Figure generation for NCD evaluation results.

All figures are 1200x800 PNGs suitable for thesis inclusion.
Uses matplotlib with a clean, publication-ready style.
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_FIGSIZE = (12, 8)
_DPI = 150

# One-time style setup
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "figure.dpi": _DPI,
    }
)


def _save(fig, name: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")
    return path


def plot_tsne_known_vs_unknown(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    known_mask: np.ndarray | None = None,
    output_dir: str = ".",
) -> str:
    """t-SNE embedding colored by species, with known/unknown markers."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    unique_labels = sorted(set(labels[labels >= 0]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for idx, cls in enumerate(unique_labels):
        mask = labels == cls
        marker = "o" if known_mask is None or known_mask[mask].any() else "x"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            label=class_names[int(cls)] if int(cls) < len(class_names) else str(cls),
            marker=marker,
            s=30,
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_title("t-SNE: Known vs Unknown Species")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return _save(fig, "tsne_known_vs_unknown", output_dir)


def plot_oscr_curve(
    oscr_points: list[dict],
    baseline_points: list[dict] | None = None,
    model_name: str = "",
    output_dir: str = ".",
) -> str:
    """OSCR curve: OpenMax vs baseline."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    rr = [p["reject_rate"] for p in oscr_points]
    ka = [p["known_acc"] for p in oscr_points]
    ax.plot(rr, ka, "b-", linewidth=2.5, label=f"OpenMax ({model_name})")

    if baseline_points:
        br = [p["reject_rate"] for p in baseline_points]
        ba = [p["known_acc"] for p in baseline_points]
        ax.plot(br, ba, "r--", linewidth=2, label=f"Softmax baseline ({model_name})")

    ax.set_xlabel("Rejection Rate (unknown)")
    ax.set_ylabel("Known Classification Accuracy")
    ax.set_title("Open-Set Classification Rate (OSCR)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "oscr_curve", output_dir)


def plot_auroc_per_class(
    per_class: list[dict],
    mean_auroc: float = 0.0,
    output_dir: str = ".",
) -> str:
    """Bar chart: AUROC for each left-out class with mean line."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    names = [p["left_out"] for p in per_class]
    values = [p["auroc"] for p in per_class]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

    bars = ax.bar(range(len(names)), values, color=colors, width=0.6)
    ax.axhline(y=mean_auroc, color="red", linestyle="--", linewidth=2, label=f"Mean AUROC = {mean_auroc:.3f}")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC per Left-Out Class (Leave-One-Class-Out)")
    ax.set_ylim(0, 1)
    ax.legend()

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return _save(fig, "auroc_per_class", output_dir)


def plot_clustering_comparison(
    algo_metrics: dict[str, dict],
    output_dir: str = ".",
) -> str:
    """Grouped bar chart: ACC/NMI/ARI/AMI/F-measure per clustering algorithm."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    metrics_names = ["acc", "nmi", "ari", "ami", "f_measure"]
    metrics_labels = ["ACC", "NMI", "ARI", "AMI", "F"]
    algo_names = list(algo_metrics.keys())
    x = np.arange(len(metrics_names))
    width = 0.8 / len(algo_names)

    colors = plt.cm.Set2(np.linspace(0, 1, len(algo_names)))

    for i, (algo, metrics) in enumerate(algo_metrics.items()):
        values = [metrics.get(m, 0) for m in metrics_names]
        offset = (i - len(algo_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=algo, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylabel("Score")
    ax.set_title("Clustering Algorithm Comparison")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "clustering_comparison", output_dir)
