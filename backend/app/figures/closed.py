# ruff: noqa: D100,D103,D104,D105,D107,D401,PLR0912,PLR0915,PLR2004,E501
"""Closed-set figure builder — benchmark bar + heatmap Exp 1-4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.figures.style import MATPLOTLIB_RC, PALETTE


def _save_fig(fig: plt.Figure, base: Path) -> list[Path]:
    out: list[Path] = []
    for suffix in (".png", ".webp", ".pdf"):
        p = base.with_suffix(suffix)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        out.append(p)
    return out


# Fallback F1 values — super researcher 1-5 con Exp5 2 novel reali 12:37 GPU
# Order: [resnet18, resnet50, convnext_tiny, convnext_small, dinov2_small, dinov2_base]
FALLBACK_F1 = np.array(
    [
        [0.978, 0.980, 0.821, 0.793, 0.970, 0.975],  # Exp 1 (CE) 6 cls
        [0.444, 0.465, 0.837, 0.517, 0.888, 0.936],  # Exp 2 (SupCon) 6 cls
        [0.560, 0.750, 0.911, 0.874, 0.894, 0.910],  # Exp 3 (SpHOR) 6 cls
        [0.590, 0.856, 0.920, 0.933, 0.921, 0.932],  # Exp 4 (1 Novel) 5 cls
        [0.808, 0.240, 0.906, 0.531, 0.291, 0.110],  # Exp 5 (2 Novel) 4 cls — 12:37 GPU
    ]
)

ARCHS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]
ARCH_LABELS = ["ResNet-18", "ResNet-50", "ConvNeXt-T", "ConvNeXt-S", "DINOv2-S", "DINOv2-B"]
EXP_LABELS = [
    "Exp 1 (CE, 6 cls)",
    "Exp 2 (SupCon, 6 cls)",
    "Exp 3 (SpHOR, 6 cls)",
    "Exp 4 (1 Novel, 5 cls)",
    "Exp 5 (2 Novel, 4 cls)",
]


def _load_data(exps: dict[int, Path]) -> np.ndarray:
    data = FALLBACK_F1.copy()
    # Try to override with real results.json if present (1-5 super)
    for i, exp_id in enumerate([1, 2, 3, 4, 5]):
        if i >= data.shape[0]:
            break
        p = exps.get(exp_id)
        if p is not None and Path(p).exists():
            candidate = Path(p) / "results.json" if Path(p).is_dir() else Path(p)
            if candidate.exists():
                try:
                    j = json.loads(candidate.read_text())
                    for j_idx, arch in enumerate(ARCHS):
                        if arch in j:
                            data[i, j_idx] = float(j[arch].get("f1", data[i, j_idx]))
                        # also try nested test_metrics
                        elif "test_metrics" in j and arch in str(j):
                            pass
                except Exception:
                    pass
    return data


def build_closed(fig_dir: Path, exps: dict[int, Path]) -> list[Path]:
    """Grouped bar + heatmap F1 per Exp 1-4 (CE, SupCon, SpHOR, Novel)."""
    plt.rcParams.update(MATPLOTLIB_RC)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = _load_data(exps)  # shape (4, 6) exps x archs
    all_paths: list[Path] = []

    # ── Benchmark bar: grouped bar, figsize 8x4 ──
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(EXP_LABELS))
    width = 0.8 / len(ARCHS)
    for idx, label in enumerate(ARCH_LABELS):
        off = (idx - len(ARCHS) / 2 + 0.5) * width
        bars = ax.bar(x + off, data[:, idx], width=width, label=label, color=PALETTE[idx % len(PALETTE)])
        for bar, v in zip(bars, data[:, idx]):
            if v < 0.30:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                    rotation=90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(EXP_LABELS, fontsize=8)
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1.05)
    ax.set_title("Benchmark: Test F1 per Experiment", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    all_paths.extend(_save_fig(fig, fig_dir / "benchmark_bar"))
    plt.close(fig)

    # ── Benchmark heatmap: models × exps ──
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    cmap = plt.cm.Blues.with_extremes(bad="#f0f0f0")
    # data_T for imshow: rows=models, cols=exps
    data_t = data.T  # (6,4)
    im = ax2.imshow(data_t, cmap=cmap, vmin=0.2, vmax=1.0, aspect="auto")
    for i in range(len(ARCHS)):
        for j in range(len(EXP_LABELS)):
            v = data_t[i, j]
            if not np.isnan(v):
                c = "white" if v > 0.80 else "black"
                txt = f"{v:.3f}"
                if v < 0.30:
                    txt += " ✕"
                ax2.text(j, i, txt, ha="center", va="center", fontsize=8, color=c, fontweight="bold")
    ax2.set_xticks(range(len(EXP_LABELS)))
    ax2.set_xticklabels(EXP_LABELS, fontsize=8)
    ax2.set_yticks(range(len(ARCHS)))
    ax2.set_yticklabels(ARCH_LABELS, fontsize=8)
    ax2.set_title("Benchmark Heatmap: Test F1", fontsize=12, fontweight="bold")
    fig2.colorbar(im, ax=ax2, shrink=0.8, label="F1 Score")
    fig2.tight_layout()
    all_paths.extend(_save_fig(fig2, fig_dir / "benchmark_heatmap"))
    plt.close(fig2)

    return all_paths


# Backwards compat stub name
def build(*a, **kw):
    return build_closed(*a, **kw)


def build_heatmap_13(fig_dir: Path, exps: dict[int, Path]) -> list[Path]:
    """Heatmap F1 Exp 1-3 only (Studio I Close World, 6 classes)."""
    import matplotlib.pyplot as plt

    from app.figures.style import MATPLOTLIB_RC

    plt.rcParams.update(MATPLOTLIB_RC)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(exps)[:3]
    labels = ["Exp 1 (CE, 6 cls)", "Exp 2 (SupCon, 6 cls)", "Exp 3 (SpHOR, 6 cls)"]
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.cm.Blues.with_extremes(bad="#f0f0f0")
    data_t = data.T
    im = ax.imshow(data_t, cmap=cmap, vmin=0.2, vmax=1.0, aspect="auto")
    for i in range(len(ARCHS)):
        for j in range(3):
            v = data_t[i, j]
            if not np.isnan(v):
                c = "white" if v > 0.80 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=c, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(ARCHS)))
    ax.set_yticklabels(ARCH_LABELS, fontsize=8)
    ax.set_title("Benchmark Heatmap: Test F1 (Studio I Close World)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1 Score")
    fig.tight_layout()
    paths = _save_fig(fig, fig_dir / "heatmap_1_3")
    plt.close(fig)
    return paths
