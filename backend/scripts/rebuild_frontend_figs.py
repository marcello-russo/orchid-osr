"""Sync public/figures dai _figures freschi + rebuild orfani legacy."""
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

os.chdir(BACKEND)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.perfold import perfold_means  # noqa: E402

DOCS = BACKEND.parent / "documents"
FIG = DOCS / "_figures"
PUB = BACKEND.parent / "frontend" / "public" / "figures"

ARCHS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]
LABELS = ["ResNet-18", "ResNet-50", "ConvNeXt-T", "ConvNeXt-S", "DINOv2-S", "DINOv2-B"]
PFX = {
    "exp1": "exp1_ce",
    "exp2": "exp2_supcon",
    "exp3": "exp3_sphor",
    "exp4": "exp4_sphor_ncd",
}


def means(exp):
    d = perfold_means(PFX[exp])
    return [d.get(m, (float("nan"), 0.0))[0] for m in ARCHS]


def build_summary_dual():
    """Come il legacy ma con F1 medi 5-fold e detection rate verificati."""
    import json

    e1, e2, e3, e4 = (means(e) for e in ["exp1", "exp2", "exp3", "exp4"])
    det = []
    for m in ARCHS:
        n = json.load(open(BACKEND / "experiments" / f"exp4_sphor_ncd_{m}" / "ncd_results.json"))[
            "detectors"
        ]
        best = max(n.items(), key=lambda kv: kv[1]["auroc"])
        det.append(best[1]["detection_rate_at_10pct_fpr"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    left = np.array([e1, e2, e3, e4]).T
    im = a1.imshow(left, cmap=plt.cm.Blues, vmin=0.2, vmax=1.0, aspect="auto")
    for i in range(6):
        for j in range(4):
            v = left[i, j]
            a1.text(
                j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                color="white" if v > 0.75 else "black", fontweight="bold",
            )
    a1.set_xticks(range(4))
    a1.set_xticklabels(["Exp 1 (CE)", "Exp 2 (SupCon)", "Exp 3 (SpHOR)", "Exp 4 (Novel)"])
    a1.set_yticks(range(6))
    a1.set_yticklabels(LABELS)
    a1.set_title("Closed-Set Test F1 (medie 5 fold)")
    d = np.array(det).reshape(-1, 1)
    a2.imshow(d, cmap=plt.cm.Blues, vmin=0, vmax=1.0, aspect="auto")
    for i, v in enumerate(det):
        a2.text(
            0, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
            color="white" if v > 0.5 else "black", fontweight="bold",
        )
    a2.set_xticks([0])
    a2.set_xticklabels(["Exp 4 detection@10%"])
    a2.set_yticks(range(6))
    a2.set_yticklabels(LABELS)
    a2.set_title("OSR Detection@10%FPR (miglior rivelatore)")
    fig.suptitle("Summary: Closed-Set and Open-World Performance", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "summary_dual_heatmap.png", dpi=150)
    plt.close(fig)
    print("summary_dual_heatmap.png rebuilt")


def build_exp9_auroc():
    """Stesse barre AUROC correnti, titolo corretto Exp 4."""
    import json

    dets = ["Mahalanobis", "KNN", "Energy", "Ensemble"]
    x = np.arange(len(ARCHS))
    w = 0.18
    cols = ["#66c2a5", "#8da0cb", "#a6d854", "#e5c07b"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, det in enumerate(dets):
        vals = []
        for m in ARCHS:
            n = json.load(open(BACKEND / "experiments" / f"exp4_sphor_ncd_{m}" / "ncd_results.json"))[
                "detectors"
            ]
            vals.append(n[det]["auroc"])
        ax.bar(x + (j - 1.5) * w, vals, w, label=det, color=cols[j])
    ax.axhline(0.5, color="gray", linestyle="--", label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Exp 4 (SpHOR, novel O. garganica): OSR AUROC", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG / "exp9_osr_auroc.png", dpi=150)
    plt.close(fig)
    print("exp9_osr_auroc.png rebuilt (titolo Exp 4)")


def to_webp(src: Path, dst: Path):
    Image.open(src).save(dst, "WEBP", quality=88)


if __name__ == "__main__":
    build_summary_dual()
    build_exp9_auroc()
    # loss curves v2 = copia fresca dal builder tesi (titolo Exp 1 corretto)
    from shutil import copy2

    copy2(FIG / "loss_curves_exp6_dinov2b.png", FIG / "loss_curves_exp6_dinov2b_v2.png")
    # sync tutto il necessario in public/figures (png + webp)
    synced = []
    for p in FIG.glob("*.png"):
        copy2(p, PUB / p.name)
        synced.append(p.name)
        to_webp(PUB / p.name, PUB / p.with_suffix(".webp").name)
    print(f"synced {len(synced)} png (+webp) -> public/figures")
