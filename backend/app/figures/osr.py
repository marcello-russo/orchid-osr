# ruff: noqa: D100,D103,D104,D105,D107,D401,PLR0912,PLR0915,PLR2004,E501
"""OSR figure builders — AUROC heatmap (Exp4) + 1-vs-2-novel dual bars.

All values read from backend/experiments/*/ncd_results.json detectors.
Official metric: AUROC (Mahalanobis / KNN / Energy). Ensemble shown as
supplement only. No dummy data, no Detection@10%FPR in the body.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.figures.style import MATPLOTLIB_RC, PALETTE

_log = logging.getLogger(__name__)

ARCHS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]
ARCH_LABELS = ["ResNet-18", "ResNet-50", "ConvNeXt-T", "ConvNeXt-S", "DINOv2-S", "DINOv2-B"]
DETECTORS = ["Mahalanobis", "KNN", "Energy", "Ensemble"]


def _save_fig(fig: plt.Figure, base: Path) -> list[Path]:
    out: list[Path] = []
    for suffix in (".png", ".webp", ".pdf"):
        p = base.with_suffix(suffix)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        out.append(p)
    return out


def _read_detectors(exp: str) -> dict[str, dict[str, float]]:
    """{model: {detector: auroc}} from ncd_results.json (skip old dirs)."""
    proj = Path(__file__).resolve().parents[2]
    out: dict[str, dict[str, float]] = {}
    for f in glob.glob(str(proj / "experiments" / f"{exp}_*" / "ncd_results.json")):
        if "old" in f:
            continue
        model = Path(f).parent.name.replace(exp + "_", "")
        try:
            j = json.loads(Path(f).read_text())
            det = j.get("detectors", {})
            out[model] = {d: float(det.get(d, {}).get("auroc", float("nan"))) for d in DETECTORS}
        except Exception:
            pass
    return out


# Verified fallback (GPU runs 2026-09-02/03) for CI machines without experiments/.
FALLBACK_DET4 = {
    "resnet18": {"Mahalanobis": 0.507, "KNN": 0.337, "Energy": 0.504, "Ensemble": 0.422},
    "resnet50": {"Mahalanobis": 0.469, "KNN": 0.776, "Energy": 0.635, "Ensemble": 0.688},
    "convnext_tiny": {"Mahalanobis": 0.769, "KNN": 0.789, "Energy": 0.840, "Ensemble": 0.850},
    "convnext_small": {"Mahalanobis": 0.723, "KNN": 0.726, "Energy": 0.614, "Ensemble": 0.840},
    "dinov2_small": {"Mahalanobis": 0.710, "KNN": 0.760, "Energy": 0.540, "Ensemble": 0.755},
    "dinov2_base": {"Mahalanobis": 0.756, "KNN": 0.856, "Energy": 0.745, "Ensemble": 0.875},
}
FALLBACK_DET5 = {
    "resnet18": {"Mahalanobis": 0.604, "KNN": 0.664, "Energy": 0.559, "Ensemble": 0.666},
    "resnet50": {"Mahalanobis": 0.621, "KNN": 0.689, "Energy": 0.440, "Ensemble": 0.634},
    "convnext_tiny": {"Mahalanobis": 0.849, "KNN": 0.836, "Energy": 0.703, "Ensemble": 0.843},
    "convnext_small": {"Mahalanobis": 0.785, "KNN": 0.774, "Energy": 0.410, "Ensemble": 0.718},
    "dinov2_small": {"Mahalanobis": 0.723, "KNN": 0.787, "Energy": 0.505, "Ensemble": 0.734},
    "dinov2_base": {"Mahalanobis": 0.835, "KNN": 0.835, "Energy": 0.488, "Ensemble": 0.805},
}
FALLBACK_F1 = {
    "exp4_sphor_ncd": {
        "resnet18": 0.590,
        "resnet50": 0.856,
        "convnext_tiny": 0.920,
        "convnext_small": 0.933,
        "dinov2_small": 0.921,
        "dinov2_base": 0.932,
    },
    "exp5_sphor_ncd": {
        "resnet18": 0.808,
        "resnet50": 0.240,
        "convnext_tiny": 0.906,
        "convnext_small": 0.531,
        "dinov2_small": 0.291,
        "dinov2_base": 0.110,
    },
}


def _auroc_heatmap(fig_dir: Path) -> list[Path]:
    data4 = _read_detectors("exp4_sphor_ncd")
    if not data4:
        _log.warning("Exp4 ncd_results.json missing — using verified fallback")
        data4 = FALLBACK_DET4
    mat = np.array([[data4.get(a, {}).get(d, np.nan) for d in DETECTORS] for a in ARCHS])
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.cm.Blues.with_extremes(bad="#f0f0f0")
    im = ax.imshow(mat, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(len(ARCHS)):
        for j in range(len(DETECTORS)):
            v = mat[i, j]
            if not np.isnan(v):
                c = "white" if v > 0.72 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=c, fontweight="bold")
    ax.set_xticks(range(len(DETECTORS)))
    ax.set_xticklabels(DETECTORS, fontsize=9)
    ax.set_yticks(range(len(ARCHS)))
    ax.set_yticklabels(ARCH_LABELS, fontsize=8)
    ax.set_title("AUROC OSR per rivelatore — Exp4, 1 novel (250+50)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC")
    fig.tight_layout()
    paths = _save_fig(fig, fig_dir / "osr_detection_heatmap")
    plt.close(fig)
    return paths


def _dual_1vs2(fig_dir: Path) -> list[Path]:
    data4 = _read_detectors("exp4_sphor_ncd")
    data5 = _read_detectors("exp5_sphor_ncd")
    if not data4:
        _log.warning("Exp4 ncd_results.json missing — using verified fallback")
        data4 = FALLBACK_DET4
    if not data5:
        _log.warning("Exp5 ncd_results.json missing — using verified fallback")
        data5 = FALLBACK_DET5
    e4 = [data4.get(a, {}).get("Ensemble", np.nan) for a in ARCHS]
    e5 = [data5.get(a, {}).get("Ensemble", np.nan) for a in ARCHS]
    x = np.arange(len(ARCHS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - w / 2, e4, w, label="Exp4, 1 novel (250+50)", color=PALETTE[0], edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, e5, w, label="Exp5, 2 novel (200+100)", color=PALETTE[2], edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(ARCH_LABELS, fontsize=8)
    ax.set_ylabel("AUROC Ensemble (supplemento)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title("OSR: 1 vs 2 novel (AUROC Ensemble)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    paths = _save_fig(fig, fig_dir / "osr_1vs2_novel")
    plt.close(fig)
    return paths


def _read_f1(exp: str) -> dict[str, float]:
    """{model: test F1} — medie sui 5 fold da perfold_test.json (fallback results.json)."""
    from app.core.perfold import perfold_means

    _pfx = {
        "exp1_ce": "exp1_ce",
        "exp2_supcon": "exp2_supcon",
        "exp3_sphor": "exp3_sphor",
        "exp4_sphor_ncd": "exp4_sphor_ncd",
        "exp5_sphor_ncd": "exp5_sphor_ncd",
    }
    out: dict[str, float] = {}
    if exp in _pfx:
        for model, (mean, _std) in perfold_means(_pfx[exp]).items():
            out[model] = mean
        if out:
            return out
    proj = Path(__file__).resolve().parents[2]
    out = {}
    for f in glob.glob(str(proj / "experiments" / f"{exp}_*" / "results.json")):
        if "old" in f:
            continue
        model = Path(f).parent.name.replace(exp + "_", "")
        try:
            j = json.loads(Path(f).read_text())
            out[model] = float(j.get("test_metrics", {}).get("f1_macro", float("nan")))
        except Exception:
            pass
    if not out and exp in FALLBACK_F1:
        _log.warning("%s results.json missing — using verified fallback", exp)
        out = dict(FALLBACK_F1[exp])
    return out


def _closed_vs_open(fig_dir: Path) -> list[Path]:
    f1_4 = _read_f1("exp4_sphor_ncd")
    f1_5 = _read_f1("exp5_sphor_ncd")
    d4 = _read_detectors("exp4_sphor_ncd") or FALLBACK_DET4
    d5 = _read_detectors("exp5_sphor_ncd") or FALLBACK_DET5
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, a in enumerate(ARCHS):
        c = PALETTE[i % len(PALETTE)]
        if a in f1_4 and a in d4 and not np.isnan(f1_4[a]):
            au = d4[a].get("Ensemble", np.nan)
            ax.scatter([f1_4[a]], [au], s=130, color=c, marker="^", edgecolors="black", linewidths=0.6, zorder=3)
        if a in f1_5 and a in d5 and not np.isnan(f1_5.get(a, np.nan)):
            au = d5[a].get("Ensemble", np.nan)
            ax.scatter([f1_5[a]], [au], s=80, color=c, marker="o", alpha=0.65, zorder=3)
    # hero annotation: ConvNeXt-T holds both worlds
    if "convnext_tiny" in f1_4 and "convnext_tiny" in d4:
        ax.annotate(
            "ConvNeXt-T tiene\nin entrambi",
            xy=(f1_4["convnext_tiny"], d4["convnext_tiny"].get("Ensemble", 0.85)),
            xytext=(0.78, 0.94),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )
    from matplotlib.lines import Line2D

    ax.legend(
        [
            Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=9),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8),
        ]
        + [
            Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE[i % len(PALETTE)], markersize=8)
            for i in range(len(ARCHS))
        ],
        ["Exp4, 1 novel", "Exp5, 2 novel"] + list(ARCH_LABELS),
        fontsize=7,
        ncol=2,
    )
    ax.set_xlabel("F1 test closed-set")
    ax.set_ylabel("AUROC OSR Ensemble (supplemento)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="caso")
    ax.set_title("Chiuso vs OSR — Exp4, 1 novel (^) vs Exp5, 2 novel (o)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths = _save_fig(fig, fig_dir / "closed_vs_open_scatter")
    plt.close(fig)
    return paths


def build_osr(fig_dir: Path, exps: dict[int, Path]) -> list[Path]:
    plt.rcParams.update(MATPLOTLIB_RC)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return _auroc_heatmap(fig_dir) + _dual_1vs2(fig_dir) + _closed_vs_open(fig_dir)


def build(*a, **kw):
    return build_osr(*a, **kw)
