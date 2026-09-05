"""NCD cross-model comparison table — importable logic (CLI: `orchid ncd-comparison`)."""

from __future__ import annotations

import json
import math
import os

EXPERIMENTS = [
    ("exp3_sphor", "Exp 3 (SpHOR)"),
    ("exp4_sphor_ncd", "Exp 4 (SpHOR-NCD)"),
    ("exp5_sphor_ncd", "Exp 5 (SpHOR-NCD 2 novel)"),
]

MODELS = [
    "resnet18",
    "resnet50",
    "convnext_tiny",
    "convnext_small",
    "dinov2_small",
    "dinov2_base",
]

DETECTOR_NAMES = ["Ensemble", "Mahalanobis", "KNN", "Energy"]


def _get_detector_auroc(data: dict, name: str) -> float:
    """Legge AUROC per un detector specifico.

    evaluate_novel() salva 'auroc', evaluate_loo() salva 'mean_auroc'.
    """
    detectors = data.get("detectors", {})
    entry = detectors.get(name, {})
    if "error" in entry:
        return float("nan")
    for key in ("auroc", "mean_auroc"):
        if key in entry:
            val = entry[key]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return float(val)
    return float("nan")


def _extract_auroc(data: dict) -> float:
    """AUROC principale: Ensemble, poi il primo detector disponibile."""
    val = _get_detector_auroc(data, "Ensemble")
    if not math.isnan(val) and val != 0:
        return val
    for name in DETECTOR_NAMES:
        val = _get_detector_auroc(data, name)
        if not math.isnan(val) and val != 0:
            return val
    openset = data.get("openset", {})
    if "auroc" in openset:
        return float(openset["auroc"])
    if "mean_auroc" in openset:
        return float(openset["mean_auroc"])
    return 0.0


def build_comparison(exp_base: str) -> list:
    """Aggregate NCD results. Returns entries (also saved to ncd_comparison.json)."""
    entries = []
    for prefix, label in EXPERIMENTS:
        for model in MODELS:
            exp_name = f"{prefix}_{model}"
            ncd_path = os.path.join(exp_base, exp_name, "ncd_results.json")
            if not os.path.exists(ncd_path):
                continue
            with open(ncd_path) as f:
                data = json.load(f)

            clustering = data.get("clustering", {})
            kmeans = clustering.get("kmeans_best", {})
            auroc = _extract_auroc(data)
            baseline = 0.0

            entry = {
                "experiment": label,
                "exp_prefix": prefix,
                "model": model,
                "selected_k": clustering.get("selected_k", 0),
                "acc": round(kmeans.get("acc", 0), 4),
                "nmi": round(kmeans.get("nmi", 0), 4),
                "ari": round(kmeans.get("ari", 0), 4),
                "ami": round(kmeans.get("ami", 0), 4),
                "f_measure": round(kmeans.get("f_measure", 0), 4),
                "silhouette": round(kmeans.get("silhouette", 0), 4),
                "mean_auroc": round(auroc, 4),
                "baseline_auroc": round(baseline, 4),
                "improvement": round(auroc - baseline, 4),
                "knn_auroc": 0,
            }

            # Per-detector AUROC per frontend
            for det_name in DETECTOR_NAMES:
                val = _get_detector_auroc(data, det_name)
                key = f"{det_name.lower()}_auroc"
                entry[key] = round(val, 4) if not math.isnan(val) else 0.0

            # Cluster AUROC per kmeans_true_k
            kmeans_k = clustering.get("kmeans_true_k", {})
            if "acc" in kmeans_k:
                entry["cluster_acc"] = round(kmeans_k["acc"], 4)
            if "nmi" in kmeans_k:
                entry["cluster_nmi"] = round(kmeans_k["nmi"], 4)

            entries.append(entry)
            print(f"  {exp_name}: AUROC={auroc:.4f}")

    entries.sort(key=lambda e: (e["experiment"], -e.get("mean_auroc", 0)))
    out_path = os.path.join(exp_base, "ncd_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"models": entries}, f, indent=2)
    print(f"\nSaved {len(entries)} entries to {out_path}")
    return entries


def main() -> None:
    from app.cli.definitions import PROJECT_ROOT

    build_comparison(str(PROJECT_ROOT / "experiments"))


if __name__ == "__main__":
    main()
