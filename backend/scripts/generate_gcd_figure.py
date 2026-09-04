#!/usr/bin/env python3
"""GCD figure: ACC Old/New per architettura — SS-KMeans (Vaze'22) vs K-means non informato.

Output: documents/_figures/gcd_old_new.png (600 DPI)
Dati: experiments/gcd_summary.json (prodotto da scripts/gcd_analysis.py)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
SUMMARY = BACKEND / "experiments" / "gcd_summary.json"
FIGURES = BACKEND.parent / "documents" / "_figures"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-T",
    "convnext_small": "ConvNeXt-S",
    "dinov2_small": "DINOv2-S",
    "dinov2_base": "DINOv2-B",
}

# palette colorblind-safe coerente col resto della tesi
C_OLD_SS = "#0072B2"  # blu pieno      — SS-KM old
C_NEW_SS = "#D55E00"  # arancio scuro  — SS-KM new
C_OLD_KM = "#56B4E9"  # azzurro        — KMeans old
C_NEW_KM = "#E69F00"  # arancio chiaro — KMeans new

data = json.load(open(SUMMARY))

old_ss = [data[m]["ss_kmeans"]["acc_old_test"] for m in MODELS]
new_ss = [data[m]["ss_kmeans"]["acc_new_test"] for m in MODELS]
old_km = [data[m]["kmeans"]["acc_old_mean"] for m in MODELS]
new_km = [data[m]["kmeans"]["acc_new_mean"] for m in MODELS]
new_km_err = [data[m]["kmeans"]["acc_new_std"] for m in MODELS]

x = np.arange(len(MODELS))
w = 0.19

fig, ax = plt.subplots(figsize=(10, 5.2))
b1 = ax.bar(x - 1.5 * w, old_ss, w, label="Old · SS-KM (Vaze'22)", color=C_OLD_SS, edgecolor="white", linewidth=0.6)
b2 = ax.bar(x - 0.5 * w, new_ss, w, label="New · SS-KM", color=C_NEW_SS, edgecolor="white", linewidth=0.6)
b3 = ax.bar(x + 0.5 * w, old_km, w, label="Old · K-means", color=C_OLD_KM, edgecolor="white", linewidth=0.6)
b4 = ax.bar(
    x + 1.5 * w,
    new_km,
    w,
    yerr=new_km_err,
    capsize=2.5,
    label="New · K-means (10 seed)",
    color=C_NEW_KM,
    edgecolor="white",
    linewidth=0.6,
)

for bars in (b1, b2, b3, b4):
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, color="#333333")

ax.axhline(1 / 6, color="#888888", linestyle=":", linewidth=1.2)
ax.text(len(MODELS) - 0.45, 1 / 6 + 0.012, "caso (1/6)", fontsize=9, color="#666666", ha="right")

ax.set_ylim(0, 1.02)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10.5)
ax.set_ylabel("ACC su pool di test (300)")
ax.set_title("GCD su Exp 4 — recupero classi note (Old) vs scoperta novel (New)", fontweight="bold", pad=34)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncols=4, fontsize=9.5)

fig.savefig(FIGURES / "gcd_old_new.png")
print(f"salvato: {FIGURES / 'gcd_old_new.png'}")
