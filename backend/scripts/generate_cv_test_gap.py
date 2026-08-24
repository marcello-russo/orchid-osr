#!/usr/bin/env python3
"""Generate cv_test_gap bar chart: (Test F1 - CV F1) per model.

Compares Exp 6 (CE) and Exp 7 (SupCon).
Output: frontend/public/figures/cv_test_gap.webp
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent / "experiments"
OUTPUT = Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "figures" / "cv_test_gap.webp"

MODELS = ["ResNet-18", "ResNet-50", "ConvNeXt-Tiny", "ConvNeXt-Small", "DINOv2-Small", "DINOv2-Base"]
EXP6_DIRS = [
    "exp6_clean_split_resnet18",
    "exp6_clean_split_resnet50",
    "exp6_clean_split_convnext_tiny",
    "exp6_clean_split_convnext_small",
    "exp6_clean_split_dinov2_small",
    "exp6_clean_split_dinov2_base",
]
EXP7_DIRS = [
    "exp7_supcon_resnet18",
    "exp7_supcon_resnet50",
    "exp7_supcon_convnext_tiny",
    "exp7_supcon_convnext_small",
    "exp7_supcon_dinov2_small",
    "exp7_supcon_dinov2_base",
]


def get_gap(exp_dir: str) -> float:
    path = BASE / exp_dir / "results.json"
    if not path.exists():
        print(f"WARNING: {path} not found, using 0.0")
        return 0.0
    data = json.loads(path.read_text())
    return data["test_metrics"]["f1_macro"] - data["mean_f1"]


def main() -> None:
    gaps_exp6 = [get_gap(d) for d in EXP6_DIRS]
    gaps_exp7 = [get_gap(d) for d in EXP7_DIRS]

    x = np.arange(len(MODELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, gaps_exp6, width, label="Exp 6 (CE)", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, gaps_exp7, width, label="Exp 7 (SupCon)", color="#DD8452")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Gap (F1 Test \u2212 F1 CV)")
    ax.set_title("Generalization Gap by Model and Paradigm")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            offset = 3 if h >= 0 else -12
            ax.annotate(
                f"{h:+.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
