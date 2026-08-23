#!/usr/bin/env python3
"""Generate all publication-quality figures for the thesis experiments chapter.

Usage:
    python scripts/generate_thesis_figures.py

Output: documents/_figures/  (relative to backend root)

All figures: 600 DPI, white background, colorblind-safe palette, seaborn style.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# ── Paths ──────────────────────────────────────────────────────────────
BACKEND = Path(__file__).resolve().parent.parent
EXPERIMENTS = BACKEND / "experiments"
FIGURES = BACKEND.parent / "documents" / "_figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Global constants ───────────────────────────────────────────────────
PATIENCE = 15
NOVEL_CLASS = "O. garganica"

MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]
EXPS_ALL = ["exp6", "exp7", "exp8", "exp9"]
EXPS_OSR = ["exp9"]

MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-T",
    "convnext_small": "ConvNeXt-S",
    "dinov2_small": "DINOv2-S",
    "dinov2_base": "DINOv2-B",
}
EXP_LABELS = {
    "exp6": "Exp 6 (CE)",
    "exp7": "Exp 7 (SupCon)",
    "exp8": "Exp 8 (SpHOR)",
    "exp9": "Exp 9 (Novel)",
}
EXP_PREFIXES = {
    "exp6": "exp6_clean_split",
    "exp7": "exp7_supcon",
    "exp8": "exp8_sphor",
    "exp9": "exp9_ncd",
}

# Colorblind-friendly palette (ibm / tableau-inspired)
CB = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#F0E442"]
SPECIES_COLORS = {
    "O. exaltata": CB[0],
    "O. garganica": CB[1],
    "O. incubacea": CB[2],
    "O. majellensis": CB[3],
    "O. sphegodes": CB[4],
    "O. sphegodes_Palena": CB[5],
}
SPECIES_SHORT = {
    "O. exaltata": "EXA",
    "O. garganica": "GAR",
    "O. incubacea": "INC",
    "O. majellensis": "MAJ",
    "O. sphegodes": "SPH",
    "O. sphegodes_Palena": "PAL",
    NOVEL_CLASS: "GAR",
}
MODEL_COLORS = {
    "resnet18": "#E69F00",
    "resnet50": "#D55E00",
    "convnext_tiny": "#009E73",
    "convnext_small": "#56B4E9",
    "dinov2_small": "#0072B2",
    "dinov2_base": "#CC79A7",
}
FOLD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# ── Matplotlib style ──────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.color": "#cccccc",
    }
)


# ── Helpers ────────────────────────────────────────────────────────────


def _save(fig, name):
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}  ({path.stat().st_size // 1024} KB)")


def short_model(m):
    return MODEL_LABELS.get(m, m)


def short_species(name):
    return SPECIES_SHORT.get(name, name)


def load_results(exp, model):
    p = EXPERIMENTS / f"{EXP_PREFIXES[exp]}_{model}" / "results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_folds(exp, model):
    pfx = EXP_PREFIXES[exp]
    folds = []
    for i in range(5):
        p = EXPERIMENTS / f"{pfx}_{model}" / f"fold_{i}.json"
        if p.exists():
            with open(p) as f:
                folds.append(json.load(f))
    return folds


def get_test_f1(res):
    """Extract test F1 from a results dict, regardless of key naming."""
    if not res:
        return 0.0
    tm = res.get("test_metrics")
    if not tm:
        return res.get("test_f1", 0.0)
    # Check different key names
    for key in ["f1", "macro_f1", "f1_macro"]:
        val = tm.get(key)
        if val is not None:
            return val
    f1pc = tm.get("f1_per_class", {})
    if f1pc:
        return float(np.mean(list(f1pc.values())))
    return 0.0


def get_detection_rate(exp, model):
    """Get Ensemble detection_rate_at_10pct_fpr from NCD results."""
    p = EXPERIMENTS / f"{EXP_PREFIXES[exp]}_{model}" / "ncd_results.json"
    if not p.exists():
        return 0.0
    with open(p) as f:
        d = json.load(f)
    dets = d.get("detectors", {})
    ens = dets.get("Ensemble", {})
    if isinstance(ens, dict):
        return ens.get("detection_rate_at_10pct_fpr", 0.0)
    return float(ens) if isinstance(ens, (int, float)) else 0.0


def trim_fold(fold):
    """Trim history to best_epoch + PATIENCE."""
    hist = dict(fold["history"])
    be = fold.get("best_epoch", len(hist.get("train_loss", [])))
    cutoff = min(be + PATIENCE, len(hist.get("train_loss", [])))
    return {k: v[:cutoff] for k, v in hist.items()}


def load_embeddings(exp, model):
    """Load raw embeddings + labels, computing t-SNE on the fly if needed."""
    ed = EXPERIMENTS / f"{EXP_PREFIXES[exp]}_{model}" / "embeddings"
    emb_file = ed / "embeddings.npy"
    lab_file = ed / "labels.npy"
    if not (emb_file.exists() and lab_file.exists()):
        return None, None, None, None

    embeddings = np.load(emb_file)
    labels = np.load(lab_file)
    meta = {}
    meta_file = ed / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
    class_names = meta.get("class_names", [])

    # Try loading precomputed t-SNE
    tsne_file = ed / "tsne_2d.npy"
    if tsne_file.exists():
        tsne = np.load(tsne_file)
    else:
        print(f"    Computing t-SNE for {exp}/{model} ({embeddings.shape[0]} pts)...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, embeddings.shape[0] - 1),
            init="random",
            learning_rate="auto",
        ).fit_transform(embeddings)

    return embeddings, labels, tsne, class_names


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 1: Benchmark Heatmap
# ══════════════════════════════════════════════════════════════════════


def fig_benchmark_heatmap():
    """Test F1 heatmap: models × experiments."""
    data = np.full((len(MODELS), len(EXPS_ALL)), np.nan)
    for j, exp in enumerate(EXPS_ALL):
        for i, model in enumerate(MODELS):
            data[i, j] = get_test_f1(load_results(exp, model))

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f0f0f0")
    im = ax.imshow(data, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")

    for i in range(len(MODELS)):
        for j in range(len(EXPS_ALL)):
            v = data[i, j]
            if not np.isnan(v):
                c = "white" if v > 0.75 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color=c, fontweight="bold")

    ax.set_xticks(range(len(EXPS_ALL)))
    ax.set_xticklabels([EXP_LABELS[e] for e in EXPS_ALL], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([short_model(m) for m in MODELS], fontsize=9)
    ax.set_title("Test F1 Score — All Models × Experiments", fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1 Score")
    fig.tight_layout()
    _save(fig, "benchmark_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2: Confusion Matrices (Exp 7, 8, 10 — exp6 has no test CM)
# ══════════════════════════════════════════════════════════════════════


def fig_confusion_matrices():
    """Confusion matrices for best model in each experiment (exp7/8/10 only)."""
    panels = [
        ("exp7", "dinov2_small", "Exp 7 (SupCon)\nDINOv2-Small"),
        ("exp8", "convnext_tiny", "Exp 8 (SpHOR)\nConvNeXt-Tiny"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for j, (exp, model, title) in enumerate(panels):
        ax = axes[j]
        res = load_results(exp, model)
        if not res or "test_metrics" not in res:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontweight="bold", fontsize=10)
            continue

        cm = np.array(res["test_metrics"]["confusion_matrix"])
        labels_raw = res["test_metrics"].get(
            "confusion_matrix_labels", list(res.get("class_names", [f"C{i}" for i in range(len(cm))]))
        )
        short = [short_species(label) for label in labels_raw]

        row_sums = cm.sum(axis=1)
        is_trainset = any(rs > 100 for rs in row_sums)

        ax.imshow(cm if not is_trainset else np.zeros_like(cm), cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontweight="bold", fontsize=10)

        for row in range(len(cm)):
            for col in range(len(cm)):
                v = cm[row, col]
                if is_trainset and row == col:
                    txt = "≥" + str(v)
                else:
                    txt = str(v)
                c_cell = "white" if (not is_trainset and v > 25) or (is_trainset and row == col) else "black"
                ax.text(col, row, txt, ha="center", va="center", fontsize=8, color=c_cell)

        ax.set_xticks(range(len(short)))
        ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.text(-0.25, -0.1, f"({chr(97 + j)})", transform=ax.transAxes, fontsize=11, fontweight="bold")

        if is_trainset:
            ax.text(
                0.5,
                -0.25,
                "Note: Exp 6 stored training-set CM (no test CM available)",
                transform=ax.transAxes,
                ha="center",
                fontsize=7,
                fontstyle="italic",
                color="gray",
            )

    fig.suptitle("Confusion Matrices on Test Set", fontweight="bold", y=1.08)
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 3: t-SNE Comparison (CE vs SupCon vs SpHOR, computed on fly)
# ══════════════════════════════════════════════════════════════════════


def fig_tsne_comparison():
    """t-SNE of DINOv2-Small: CE -> SupCon -> SpHOR."""
    model = "dinov2_small"
    exps = ["exp6", "exp7", "exp8"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for j, exp in enumerate(exps):
        ax = axes[j]
        _, _, tsne, cnames = load_embeddings(exp, model)
        if tsne is None:
            ax.text(0.5, 0.5, "Embeddings N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(EXP_LABELS[exp], fontweight="bold")
            ax.text(-0.15, -0.1, f"({chr(97 + j)})", transform=ax.transAxes, fontsize=11, fontweight="bold")
            continue

        labels = np.load(EXPERIMENTS / f"{EXP_PREFIXES[exp]}_{model}" / "embeddings" / "labels.npy")
        unique = sorted(np.unique(labels))
        for cls in unique:
            mask = labels == cls
            name = cnames[int(cls)] if int(cls) < len(cnames) else f"C{cls}"
            sn = short_species(name)
            c = SPECIES_COLORS.get(name, CB[int(cls) % len(CB)])
            ax.scatter(tsne[mask, 0], tsne[mask, 1], c=c, label=sn, s=12, alpha=0.6, edgecolors="none", rasterized=True)

        ax.set_title(EXP_LABELS[exp], fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=2, framealpha=0.9)
        ax.text(-0.15, -0.1, f"({chr(97 + j)})", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.suptitle("t-SNE Embedding Space — DINOv2-Small", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "tsne_comparison.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 4: Closed-Set vs Open-Set Scatter
# ══════════════════════════════════════════════════════════════════════


def fig_closed_vs_open_scatter():
    """Scatter: test F1 vs detection_rate@10%FPR. Shows exp9+10 (only ones with OSR data)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for model in MODELS:
        f1 = get_test_f1(load_results("exp9", model))
        dr = get_detection_rate("exp9", model)
        ax.scatter(
            f1,
            dr,
            s=130,
            c=MODEL_COLORS.get(model, CB[0]),
            marker="^",
            label=short_model(model),
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(short_model(model), (f1, dr), textcoords="offset points", xytext=(5, 4), fontsize=7, alpha=0.8)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(x=0.7, color="gray", linestyle=":", alpha=0.4)
    ax.text(0.72, 0.88, "HIGH F1 +\nHIGH OSR", fontsize=9, fontweight="bold", color="#0072B2", alpha=0.4)
    ax.text(0.72, 0.02, "HIGH F1 +\nLOW OSR", fontsize=9, fontweight="bold", color="#D55E00", alpha=0.4)
    ax.set_xlabel("Closed-Set Test F1 (Exp 9)")
    ax.set_ylabel("Open-Set Detection Rate @ 10% FPR (Exp 9)")
    ax.set_title("Closed-Set vs Open-World Performance — Exp 9 (novel O. garganica)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save(fig, "closed_vs_open_scatter.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 5: Per-Species F1 (best model per experiment)
# ══════════════════════════════════════════════════════════════════════


def fig_per_species_f1():
    """Grouped bar chart: F1 per species, best model per experiment."""
    best = {"exp6": "resnet50", "exp7": "dinov2_small", "exp8": "convnext_tiny"}
    order = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]
    short = [short_species(s) for s in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    n_exp, n_sp = len(best), len(order)
    width = 0.8 / n_exp
    x = np.arange(n_sp)

    for j, (exp, model) in enumerate(best.items()):
        res = load_results(exp, model)
        f1pc = res["test_metrics"].get("f1_per_class", {}) if res and "test_metrics" in res else {}
        vals = [f1pc.get(s, 0) for s in order]
        off = (j - n_exp / 2 + 0.5) * width
        bars = ax.bar(x + off, vals, width, label=EXP_LABELS[exp], color=plt.cm.Set2(j / n_exp))
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Species F1 — Best Model per Experiment", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    _save(fig, "per_species_f1.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 6: CV-Test Gap (annotated, clearer)
# ══════════════════════════════════════════════════════════════════════


def fig_cv_test_gap():
    """Scatter: model params vs CV-test F1 gap with double-descent annotation."""
    params = {
        "resnet18": 11,
        "resnet50": 25,
        "convnext_tiny": 28,
        "convnext_small": 50,
        "dinov2_small": 21,
        "dinov2_base": 86,
    }
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp in ["exp6", "exp7"]:
        gaps = []
        for model in MODELS:
            res = load_results(exp, model)
            cv = res.get("mean_f1", 0) if res else 0
            test = get_test_f1(res)
            gaps.append(test - cv)

        c = CB[0] if exp == "exp6" else CB[1]
        ax.scatter(
            [params[m] for m in MODELS],
            gaps,
            s=90,
            c=c,
            zorder=3,
            label=EXP_LABELS[exp],
            edgecolors="black",
            linewidths=0.3,
        )
        for i, m in enumerate(MODELS):
            ax.annotate(
                short_model(m), (params[m], gaps[i]), textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7
            )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(60, 0.03, "Overfits on val\n(test > CV)", fontsize=8, color="gray", fontstyle="italic")
    ax.text(60, -0.08, "Underfits on val\n(test < CV)", fontsize=8, color="gray", fontstyle="italic")
    ax.set_xlabel("Model Parameters (millions)")
    ax.set_ylabel("Test F1 − CV F1  (gap)")
    ax.set_title("Generalisation Gap vs Model Size", fontweight="bold")
    ax.legend()
    ax.set_xlim(5, 100)
    fig.tight_layout()
    _save(fig, "cv_test_gap.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 7: Loss Curves — aggregated mean±std per experiment
# ══════════════════════════════════════════════════════════════════════


def fig_loss_curves_aggregated():
    """Validation loss: one panel, one colored line per experiment.
    Each line is mean±std across folds for DINOv2-Small. Shows training dynamics
    of the three main paradigms (CE → SupCon → SpHOR → MGH-NCD)."""
    model = "dinov2_small"
    fig, ax = plt.subplots(figsize=(9, 5))
    exp_colors = {"exp6": "#757575", "exp7": "#E69F00", "exp8": "#0072B2", "exp9": "#009E73", "exp10": "#D55E00"}
    exp_styles = {"exp6": "-", "exp7": "--", "exp8": "-.", "exp9": ":", "exp10": ":"}

    for idx, exp in enumerate(EXPS_ALL):
        folds = load_folds(exp, model)
        if not folds:
            continue
        trimmed = [trim_fold(f) for f in folds]
        min_len = min(len(h["val_loss"]) for h in trimmed)
        aligned = np.array([h["val_loss"][:min_len] for h in trimmed])
        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)
        epochs = np.arange(1, min_len + 1)
        ax.plot(epochs, mean, color=exp_colors[exp], linestyle=exp_styles[exp], label=EXP_LABELS[exp], linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=exp_colors[exp], alpha=0.1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Training Dynamics — DINOv2-Small", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "loss_curves_aggregated.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 8: OSR Detection Rate — all models, Exp 8
# ══════════════════════════════════════════════════════════════════════


def fig_osr_detection_rate():
    """Detection rate @ 10% FPR per detector, all models, Exp 9 (novel = O. garganica)."""
    detectors = ["Mahalanobis", "KNN", "Energy", "Ensemble"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODELS))
    width = 0.8 / len(detectors)

    for j, det in enumerate(detectors):
        vals = []
        for model in MODELS:
            p = EXPERIMENTS / f"{EXP_PREFIXES['exp9']}_{model}" / "ncd_results.json"
            v = 0
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                dets = d.get("detectors", {})
                dd = dets.get(det, {})
                v = dd.get("detection_rate_at_10pct_fpr", 0) if isinstance(dd, dict) else 0
            vals.append(v)
        off = (j - len(detectors) / 2 + 0.5) * width
        ax.bar(x + off, vals, width, label=det, color=plt.cm.Set2(j / len(detectors)))

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in MODELS], rotation=30, ha="right")
    ax.set_ylabel("Detection Rate @ 10% FPR")
    ax.set_title("Exp 9 (SpHOR, novel O. garganica): Detection @ 10% FPR", fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "osr_detection_rate.png")


def fig_exp9_osr_auroc():
    """AUROC OSR per detector, all models, Exp 9 (novel = O. garganica)."""
    detectors = ["Mahalanobis", "KNN", "Energy", "Ensemble"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODELS))
    width = 0.8 / len(detectors)

    for j, det in enumerate(detectors):
        vals = []
        for model in MODELS:
            p = EXPERIMENTS / f"{EXP_PREFIXES['exp9']}_{model}" / "ncd_results.json"
            v = 0
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                dets = d.get("detectors", {})
                dd = dets.get(det, {})
                v = dd.get("auroc", 0) if isinstance(dd, dict) else 0
            vals.append(v)
        off = (j - len(detectors) / 2 + 0.5) * width
        ax.bar(x + off, vals, width, label=det, color=plt.cm.Set2(j / len(detectors)))

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in MODELS], rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Exp 9 (SpHOR, novel O. garganica): OSR AUROC", fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance (0.5)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "exp9_osr_auroc.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 9: OSR Detection Rate — Exp 8 vs Exp 10 side-by-side
# ══════════════════════════════════════════════════════════════════════


def fig_osr_detection_comparison():
    """Detection rate: Exp 9 (novel garganica) vs Exp 10, all detectors, all models."""
    detectors = ["Mahalanobis", "KNN", "Energy", "Ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(MODELS))
    width = 0.8 / len(detectors)

    for idx, exp in enumerate(["exp9", "exp10"]):
        ax = axes[idx]
        for j, det in enumerate(detectors):
            vals = []
            for model in MODELS:
                p = EXPERIMENTS / f"{EXP_PREFIXES[exp]}_{model}" / "ncd_results.json"
                v = 0
                if p.exists():
                    with open(p) as f:
                        d = json.load(f)
                    dets = d.get("detectors", {})
                    dd = dets.get(det, {})
                    v = dd.get("detection_rate_at_10pct_fpr", 0) if isinstance(dd, dict) else 0
                vals.append(v)
            off = (j - len(detectors) / 2 + 0.5) * width
            ax.bar(x + off, vals, width, label=det, color=plt.cm.Set2(j / len(detectors)))
        ax.set_xticks(x)
        ax.set_xticklabels([short_model(m) for m in MODELS], rotation=30, ha="right")
        ax.set_ylabel("Detection Rate @ 10% FPR")
        ax.set_title(EXP_LABELS[exp], fontweight="bold")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=8)
        ax.text(-0.15, -0.1, f"({chr(97 + idx)})", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.suptitle("Known-Class Recognition Rate: Exp 9 (novel garganica) vs Exp 10", fontweight="bold", y=1.05)
    fig.tight_layout()
    _save(fig, "osr_detection_comparison.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 10: Exp 8 per-species F1 — all models
# ══════════════════════════════════════════════════════════════════════


def fig_exp8_per_species():
    """Per-species F1, all models, Exp 8."""
    order = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]
    short = [short_species(s) for s in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(MODELS)
    x = np.arange(len(order))

    for i, model in enumerate(MODELS):
        res = load_results("exp8", model)
        f1pc = res["test_metrics"].get("f1_per_class", {}) if res and "test_metrics" in res else {}
        vals = [f1pc.get(s, 0) for s in order]
        off = (i - len(MODELS) / 2 + 0.5) * width
        bars = ax.bar(x + off, vals, width, label=short_model(model), color=MODEL_COLORS.get(model, CB[i]))
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("F1 Score")
    ax.set_title("Exp 8 (SpHOR): Per-Species F1 by Architecture", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=8)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    _save(fig, "exp8_per_species.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 11: Exp 10 per-species F1 — all models
# ══════════════════════════════════════════════════════════════════════


def fig_exp10_per_species():
    """Per-species F1, all models, Exp 10."""
    order = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]
    short = [short_species(s) for s in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(MODELS)
    x = np.arange(len(order))

    for i, model in enumerate(MODELS):
        res = load_results("exp10", model)
        f1pc = res["test_metrics"].get("f1_per_class", {}) if res and "test_metrics" in res else {}
        vals = [f1pc.get(s, 0) for s in order]
        off = (i - len(MODELS) / 2 + 0.5) * width
        bars = ax.bar(x + off, vals, width, label=short_model(model), color=MODEL_COLORS.get(model, CB[i]))
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("F1 Score")
    ax.set_title("Exp 10 (MGH-NCD): Per-Species F1 by Architecture", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=8)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    _save(fig, "exp10_per_species.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 12: Exp 8 t-SNE — multiple architectures
# ══════════════════════════════════════════════════════════════════════


def fig_tsne_exp8_multimodel():
    """t-SNE for 3 architectures in Exp 8."""
    sel = [("resnet50", "ResNet-50"), ("dinov2_small", "DINOv2-Small"), ("convnext_tiny", "ConvNeXt-Tiny")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for j, (model, label) in enumerate(sel):
        ax = axes[j]
        _, _, tsne, cnames = load_embeddings("exp8", model)
        if tsne is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontweight="bold")
            continue

        labs = np.load(EXPERIMENTS / f"{EXP_PREFIXES['exp8']}_{model}" / "embeddings" / "labels.npy")
        unique = sorted(np.unique(labs))
        for cls in unique:
            mask = labs == cls
            name = cnames[int(cls)] if int(cls) < len(cnames) else f"C{cls}"
            sn = short_species(name)
            c = SPECIES_COLORS.get(name, CB[int(cls) % len(CB)])
            ax.scatter(tsne[mask, 0], tsne[mask, 1], c=c, label=sn, s=10, alpha=0.6, edgecolors="none", rasterized=True)

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=2, framealpha=0.9)
        ax.text(-0.15, -0.1, f"({chr(97 + j)})", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.suptitle("Exp 8 (SpHOR): t-SNE per Architecture", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "tsne_exp8_multimodel.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 13: Benchmark bar chart (legend outside)
# ══════════════════════════════════════════════════════════════════════


def fig_benchmark_bar():
    """Grouped bar chart: all models × all experiments."""
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / len(MODELS)
    x = np.arange(len(EXPS_ALL))

    for i, model in enumerate(MODELS):
        vals = [get_test_f1(load_results(exp, model)) for exp in EXPS_ALL]
        off = (i - len(MODELS) / 2 + 0.5) * width
        ax.bar(x + off, vals, width, label=short_model(model), color=MODEL_COLORS.get(model, CB[i]))

    ax.set_xticks(x)
    ax.set_xticklabels([EXP_LABELS[e] for e in EXPS_ALL])
    ax.set_ylabel("Test F1 Score")
    ax.set_title("Benchmark: All Models × All Experiments", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=10)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "benchmark_bar.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 14: Clustering comparison (SpHOR vs MGH-NCD)
# ══════════════════════════════════════════════════════════════════════


def fig_clustering_comparison():
    """Clustering metrics: SpHOR vs MGH-NCD for DINOv2-Base."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["acc", "nmi", "ari", "ami"]
    metric_labels = ["ACC", "NMI", "ARI", "AMI"]
    exps = [("exp8", "SpHOR"), ("exp10", "MGH-NCD")]
    model = "dinov2_base"
    x = np.arange(len(metrics))
    width = 0.35

    for j, (exp, label) in enumerate(exps):
        p = EXPERIMENTS / f"{EXP_PREFIXES[exp]}_{model}" / "ncd_results.json"
        vals = [0] * len(metrics)
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            cl = d.get("clustering", {})
            best = cl.get("kmeans_best", {})
            for i, m in enumerate(metrics):
                vals[i] = best.get(m, 0)
        off = (j - 0.5) * width
        bars = ax.bar(x + off, vals, width, label=label, color=CB[j * 3 % len(CB)])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Clustering Performance — DINOv2-Base", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "clustering_comparison.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 15: Summary dual heatmap
# ══════════════════════════════════════════════════════════════════════


def fig_summary_dual_heatmap():
    """Closed-set F1 + open-set detection rate side by side."""
    data_f1 = np.full((len(MODELS), 5), np.nan)
    data_det = np.full((len(MODELS), len(EXPS_OSR)), np.nan)

    for j, exp in enumerate(EXPS_ALL):
        for i, model in enumerate(MODELS):
            data_f1[i, j] = get_test_f1(load_results(exp, model))

    for j, exp in enumerate(EXPS_OSR):
        for i, model in enumerate(MODELS):
            data_det[i, j] = get_detection_rate(exp, model)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f0f0f0")

    for ax, data, title, vmin, vmax, labels in [
        (axes[0], data_f1, "Closed-Set Test F1", 0.3, 1.0, [EXP_LABELS[e] for e in EXPS_ALL]),
        (axes[1], data_det, "OSR Detection@10%FPR", 0.0, 1.0, [EXP_LABELS[e] for e in EXPS_OSR]),
    ]:
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([short_model(m) for m in MODELS], fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=10)
        for i in range(len(MODELS)):
            for jj in range(data.shape[1]):
                v = data[i, jj]
                if not np.isnan(v):
                    thr = 0.6 if title.startswith("OSR") else 0.7
                    c = "white" if v > thr else "black"
                    ax.text(jj, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color=c, fontweight="bold")

    fig.suptitle("Summary: Closed-Set and Open-World Performance", fontweight="bold", y=1.05)
    fig.tight_layout()
    _save(fig, "summary_dual_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 16: OSR detection heatmap (Exp 8 vs Exp 10)
# ══════════════════════════════════════════════════════════════════════


def fig_osr_detection_heatmap():
    """Ensemble detection rate for all models, Exp 9/10."""
    data = np.full((len(MODELS), len(EXPS_OSR)), np.nan)
    for j, exp in enumerate(EXPS_OSR):
        for i, model in enumerate(MODELS):
            data[i, j] = get_detection_rate(exp, model)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f0f0f0")
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(len(MODELS)):
        for j in range(len(EXPS_OSR)):
            v = data[i, j]
            if not np.isnan(v):
                c = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=12, color=c, fontweight="bold")

    ax.set_xticks(range(len(EXPS_OSR)))
    ax.set_xticklabels(["Novel\n(Exp 9)"], fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([short_model(m) for m in MODELS], fontsize=9)
    ax.set_title("Detection Rate @ 10% FPR", fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Detection Rate")
    fig.tight_layout()
    _save(fig, "osr_detection_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
#  SPOTLIGHT 17: ConvNeXt-Tiny — the OSR champion
# ══════════════════════════════════════════════════════════════════════


def fig_spotlight_convnext_osr():
    """Spotlight: DINOv2-Base achieves AUROC 0.875 in Exp 9 (novel = O. garganica).
    Shows AUROC across ALL detectors + models, highlighting DINOv2-Base."""
    detectors = ["Mahalanobis", "KNN", "Energy", "Ensemble"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(detectors))
    width = 0.12

    for i, model in enumerate(MODELS):
        p = EXPERIMENTS / f"{EXP_PREFIXES['exp9']}_{model}" / "ncd_results.json"
        vals = [0] * len(detectors)
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            dets = d.get("detectors", {})
            for j, det in enumerate(detectors):
                dd = dets.get(det, {})
                vals[j] = dd.get("auroc", 0) if isinstance(dd, dict) else 0

        off = (i - len(MODELS) / 2 + 0.5) * width
        c = "#D55E00" if model == "dinov2_base" else MODEL_COLORS.get(model, CB[i])
        lw = 2.5 if model == "dinov2_base" else 0.8
        alpha_val = 1.0 if model == "dinov2_base" else 0.4
        ax.bar(
            x + off,
            vals,
            width,
            label=short_model(model),
            color=c,
            linewidth=lw,
            edgecolor="black" if model == "dinov2_base" else None,
            alpha=alpha_val,
        )

    # Annotate the champion
    ax.annotate(
        "DINOv2-Base: 0.875\n(Ensemble detector)",
        xy=(3 + (MODELS.index("dinov2_base") - len(MODELS) / 2 + 0.5) * width, 0.875),
        xytext=(4.5, 0.80),
        fontsize=10,
        fontweight="bold",
        color="#D55E00",
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D55E00", alpha=0.9),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(detectors)
    ax.set_ylabel("AUROC")

    ax.set_title("OSR Champion: DINOv2-Base + SpHOR (Exp 9, novel O. garganica)", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "spotlight_convnext_osr.png")


# ══════════════════════════════════════════════════════════════════════
#  SPOTLIGHT 18: Exp 10 collapse to sphegodes_Palena
# ══════════════════════════════════════════════════════════════════════


def fig_spotlight_exp10_collapse():
    """Focused figure: Exp 10 causes most species to be classified as Palena.
    Shows per-species accuracy breakdown highlighting the collapse."""
    model = "dinov2_base"
    res = load_results("exp10", model)
    cm = np.array(res["test_metrics"]["confusion_matrix"])
    labels_raw = res["test_metrics"].get(
        "confusion_matrix_labels", list(res.get("class_names", [f"C{i}" for i in range(len(cm))]))
    )
    short = [short_species(label) for label in labels_raw]
    n = len(cm)

    # Per-class accuracy (correct / total)
    row_sums = cm.sum(axis=1, keepdims=True)
    accuracy = cm.diagonal() / row_sums.flatten() if row_sums.min() > 0 else np.zeros(n)

    # How many samples of each class went to Palena (last column)
    palena_col = cm[:, -1]
    palena_frac = palena_col / row_sums.flatten() if row_sums.min() > 0 else np.zeros(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): confusion matrix heatmap
    ax = axes[0]
    ax.imshow(cm, cmap="Reds", interpolation="nearest", vmin=0, vmax=50)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            c = "white" if v > 25 else "black"
            fw = "bold" if j == n - 1 else "normal"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9, color=c, fontweight=fw)

    # Highlight the Palena column
    for i in range(n):
        rect = plt.Rectangle((n - 1 - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#D55E00", linewidth=3, linestyle="--")
        ax.add_patch(rect)

    ax.set_title("(a) Confusion Matrix — Collapse to Palena", fontweight="bold", fontsize=11)
    ax.text(-0.15, -0.1, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # Panel (b): accuracy + palena fraction
    ax = axes[1]
    x = np.arange(n)
    width = 0.35
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Correct class", color="#0072B2")
    bars2 = ax.bar(x + width / 2, palena_frac, width, label="Classified as Palena", color="#D55E00")

    for bar, v in zip(bars1, accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    for bar, v in zip(bars2, palena_frac):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#D55E00",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Fraction")
    ax.set_title("(b) Correct vs Classified-as-Palena", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.text(-0.15, -0.1, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.suptitle("Exp 10 (MGH-NCD): DINOv2-Base Collapse", fontweight="bold", y=1.05)
    fig.tight_layout()
    _save(fig, "spotlight_exp10_collapse.png")


# ══════════════════════════════════════════════════════════════════════
#  SPOTLIGHT 19: Per-species F1 evolution across experiments
# ══════════════════════════════════════════════════════════════════════


def fig_spotlight_species_evolution():
    """How per-species F1 changes across experiments for DINOv2-Small.
    Shows the journey from CE → SupCon → SpHOR → MGH-NCD for each species.
    Exp 9 is excluded: O. garganica is the novel class there (not in the
    closed-set evaluation), so its per-species F1 would be a misleading 0."""
    model = "dinov2_small"
    order = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]
    short = [short_species(s) for s in order]
    n_sp = len(order)
    exps_shown = ["exp6", "exp7", "exp8"]

    fig, ax = plt.subplots(figsize=(10, 5))
    exp_colors = {"exp6": "#757575", "exp7": "#E69F00", "exp8": "#0072B2", "exp9": "#009E73"}
    markers = {"exp6": "o", "exp7": "s", "exp8": "^", "exp10": "D"}
    x = np.arange(n_sp)

    for exp in exps_shown:
        res = load_results(exp, model)
        f1pc = res["test_metrics"].get("f1_per_class", {}) if res and "test_metrics" in res else {}
        vals = [f1pc.get(s, 0) for s in order]
        ax.plot(
            x,
            vals,
            color=exp_colors[exp],
            marker=markers[exp],
            label=EXP_LABELS[exp],
            linewidth=1.5,
            markersize=8,
            zorder=3,
        )
        # Annotate values
        for i, v in enumerate(vals):
            ax.annotate(
                f"{v:.2f}",
                (x[i], v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=6,
                color=exp_colors[exp],
                alpha=0.7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Species F1 Evolution — DINOv2-Small", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "spotlight_species_evolution.png")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("Generating thesis figures...")
    print(f"Output: {FIGURES}")
    print("=" * 60)

    todo = [
        ("1. Benchmark heatmap", fig_benchmark_heatmap),
        ("2. Benchmark bar chart", fig_benchmark_bar),
        ("3. Confusion matrices (exp7/8/10)", fig_confusion_matrices),
        ("4. t-SNE comparison (CE vs SupCon vs SpHOR)", fig_tsne_comparison),
        ("5. Per-species F1 (best models)", fig_per_species_f1),
        ("6. CV-test gap (double descent)", fig_cv_test_gap),
        ("7. Loss curves — aggregated", fig_loss_curves_aggregated),
        ("--- EXP 8 (SpHOR) ---", None),
        ("8. OSR detection rate", fig_osr_detection_rate),
        ("9. t-SNE Exp 8 — 3 architectures", fig_tsne_exp8_multimodel),
        ("10. Per-species F1 all models", fig_exp8_per_species),
        ("--- EXP 9 (Novel: O. garganica) ---", None),
        ("10b. OSR AUROC (novel garganica)", fig_exp9_osr_auroc),
        ("--- CROSS-EXP ---", None),
        ("--- SPOTLIGHT FIGURES ---", None),
        ("11. Closed vs Open scatter", fig_closed_vs_open_scatter),
        ("12. Summary dual heatmap", fig_summary_dual_heatmap),
        ("13. OSR detection heatmap", fig_osr_detection_heatmap),
        ("14. Spotlight: DINOv2-Base OSR champion", fig_spotlight_convnext_osr),
        ("15. Spotlight: per-species evolution across exps", fig_spotlight_species_evolution),
    ]

    for name, func in todo:
        print(f"\n[{name}]")
        if func is None:
            continue
        try:
            func()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback

            traceback.print_exc()

    n = len(list(FIGURES.glob("*.png")))
    print(f"\n{'=' * 60}")
    print(f"Done. {n} figures in {FIGURES}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
