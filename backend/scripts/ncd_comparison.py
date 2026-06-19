import json
import os

EXP_BASE = os.path.join(os.path.dirname(__file__), "..", "experiments")
EXP_BASE = os.path.abspath(EXP_BASE)
OUTPUT = os.path.join(EXP_BASE, "ncd_comparison.json")

EXPERIMENTS = [
    ("exp8_sphor", "Exp 8 (SpHOR)"),
    ("exp10_mgh_ncd", "Exp 10 (MGH-NCD)"),
]

MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]


def main():
    entries = []
    for prefix, label in EXPERIMENTS:
        for model in MODELS:
            exp_name = f"{prefix}_{model}"
            ncd_path = os.path.join(EXP_BASE, exp_name, "ncd_results.json")
            if not os.path.exists(ncd_path):
                continue
            with open(ncd_path) as f:
                data = json.load(f)

            clustering = data.get("clustering", {})
            kmeans = clustering.get("kmeans_best", {})
            openset = data.get("openset", {})
            baseline = openset.get("baselines", {}).get("softmax_threshold", {})
            auroc = openset.get("mean_auroc", 0)
            knn = openset.get("knn_auroc", 0)

            entries.append({
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
                "knn_auroc": round(knn, 4),
                "baseline_auroc": round(baseline.get("mean_auroc", 0), 4),
                "improvement": round(auroc - baseline.get("mean_auroc", 0), 4),
            })
            print(f"  {exp_name}: AUROC={auroc:.4f} (baseline={baseline.get('mean_auroc',0):.4f})")

    entries.sort(key=lambda e: (e["experiment"], -e.get("mean_auroc", 0)))
    with open(OUTPUT, "w") as f:
        json.dump({"models": entries}, f, indent=2)
    print(f"\nSaved {len(entries)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
