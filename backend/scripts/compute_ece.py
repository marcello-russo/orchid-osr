"""Real ECE computation for the defense reliability diagram (F2).

Loads the best fold of ConvNeXt-Tiny Exp3 (SpHOR), runs the 300-image test
set on CPU, and computes 15-bin Expected Calibration Error from softmax
confidences. Regenerates documents/_figures/reliability_diagram.png with
REAL data (the previous figure used seeded dummy data).

Usage: python scripts/compute_ece.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.core.models import create_model, create_transform  # noqa: E402
from app.preprocessing.dataset_tool import ImageFolderWithName  # noqa: E402

EXP_DIR = BACKEND / "experiments" / "exp3_sphor_convnext_tiny"
TEST_DIR = BACKEND / "dataset" / "cropped_224x224" / "test"
FIG_OUT = BACKEND.parent / "documents" / "_figures" / "reliability_diagram.png"
N_BINS = 15


def main() -> int:
    res = json.loads((EXP_DIR / "results.json").read_text())
    best_fold = res["best_fold"]
    wpath = EXP_DIR / f"best_fold_{best_fold}.pt"
    print(f"model=convnext_tiny exp=3 fold={best_fold} weights={wpath.name}")

    transform = create_transform("convnext_tiny")
    ds = ImageFolderWithName(str(TEST_DIR), transform=transform)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    print(f"test images: {len(ds)} classes: {ds.classes}")

    model = create_model("convnext_tiny", num_classes=6, pretrained=False, normalize_embeddings=True)
    model.load_weights(str(wpath))
    model = model.to("cpu")
    model.eval()

    confs, correct = [], []
    with torch.no_grad():
        for inputs, labels, _ in loader:
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            confs.extend(conf.numpy())
            correct.extend((pred.numpy() == labels.numpy()).astype(float))
    confs = np.array(confs)
    correct = np.array(correct)
    print(f"accuracy={correct.mean():.4f} (expect ~0.911)")

    edges = np.linspace(0, 1, N_BINS + 1)
    acc, cf, cnt = [], [], []
    ece = 0.0
    for i in range(N_BINS):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        n = int(m.sum())
        cnt.append(n)
        if n == 0:
            acc.append(0.0)
            cf.append((edges[i] + edges[i + 1]) / 2)
            continue
        a = float(correct[m].mean())
        c = float(confs[m].mean())
        acc.append(a)
        cf.append(c)
        ece += (n / len(confs)) * abs(a - c)
    acc = np.array(acc)
    cf = np.array(cf)
    print(f"ECE({N_BINS} bins) = {ece:.4f}")
    print("non-empty bins:", sum(1 for n in cnt if n > 0), "counts:", cnt)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.plot(cf, acc, marker="o", color="#002FA7", markersize=5, label="ConvNeXt-Tiny Exp3")
    ax.bar((edges[:-1] + edges[1:]) / 2, acc, width=0.06, alpha=0.3, color="#002FA7", edgecolor="none")
    ax.set_xlabel("Confidence (max prob)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability diagram — ECE={ece:.3f} ({N_BINS} bins)", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=200)
    print(f"saved {FIG_OUT}")
    (BACKEND / "experiments" / "exp3_sphor_convnext_tiny" / "ece.json").write_text(
        json.dumps({"ece": round(float(ece), 4), "n_bins": N_BINS, "n": len(confs),
                    "model": "convnext_tiny", "exp": 3, "fold": best_fold}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
