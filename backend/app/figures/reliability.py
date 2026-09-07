# ruff: noqa: D100,D103,D104,D105,D107,D401,PLR0912,PLR0915,PLR2004,E501
"""Reliability figure builders — reliability diagram + confusion matrix."""

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


def build_reliability(fig_dir: Path, exps: dict[int, Path]) -> list[Path]:
    plt.rcParams.update(MATPLOTLIB_RC)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    # --- reliability diagram: ONLY real data (see scripts/compute_ece.py) ---
    # Mai dati finti: se i bin reali mancano, fallire ad alta voce.
    data_path = Path(__file__).resolve().parent / "_data" / "ece_bins.json"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} mancante: generare con `python scripts/compute_ece.py`")
    bins = json.loads(data_path.read_text(encoding="utf-8"))
    n_bins = int(bins["n_bins"])
    acc = np.array(bins["acc"])
    conf = np.array(bins["conf"])
    counts = np.array(bins["counts"])
    ece = float(np.sum(counts / counts.sum() * np.abs(acc - conf)))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideale")
    ax.plot(conf, acc, marker="o", color=PALETTE[0], markersize=4, label="ConvNeXt-Tiny Exp3")
    # bars
    ax.bar(bin_centers, acc, width=0.06, alpha=0.3, color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Confidenza (prob. max)")
    ax.set_ylabel("Accuratezza")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Diagramma di affidabilità — ECE={ece:.3f} ({n_bins} bin)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    paths.extend(_save_fig(fig, fig_dir / "reliability_diagram"))
    plt.close(fig)

    return paths


def build(*a, **kw):
    return build_reliability(*a, **kw)
