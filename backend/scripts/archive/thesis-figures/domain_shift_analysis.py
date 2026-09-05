"""Domain shift analysis: field vs internet photos.

NOTE: This script's assumptions are INCORRECT for per-image source mapping.
All 6 species have field (private) photos. Only 4 have internet photos
(exaltata, garganica, incubacea, sphegodes). Majellensis and Palena are
field-only. Without per-image source labels in the test set metadata,
within-source accuracy cannot be computed reliably.

The script is kept for reference but its output should NOT be used for
conclusions about domain-specific shortcuts. The qualitative discussion
in the thesis (confusion patterns) is more reliable.
"""

import json
import os
from pathlib import Path

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "experiments")
SPECIES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. majellensis",
    "O. sphegodes",
    "O. sphegodes Palena",
]
# 0=internet, 1=field
SOURCE = [0, 0, 0, 1, 0, 1]
SRC_NAMES = {0: "internet", 1: "campo"}


def within_source_accuracy(confusion: np.ndarray, indices: list[int]) -> float:
    correct = float(sum(confusion[i, i] for i in indices))
    total = float(sum(confusion[i, :].sum() for i in indices))
    return correct / total if total > 0 else 0.0


def cross_source_rate(confusion: np.ndarray, indices_from: list[int], indices_to: list[int]) -> float:
    errors = float(sum(confusion[i, j] for i in indices_from for j in indices_to))
    total = float(sum(confusion[i, :].sum() for i in indices_from))
    return errors / total if total > 0 else 0.0


def analyze_experiment(exp_dir: str, label: str) -> dict | None:
    path = os.path.join(BASE, exp_dir, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    cm = np.array(data["test_metrics"]["confusion_matrix"])
    n = cm.shape[0]
    assert n == 6, f"Expected 6 classes, got {n}"

    # Per-class F1 from confusion matrix
    f1_per_class = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class.append(f1)

    macro_f1 = np.mean(f1_per_class)

    # Within-source accuracy
    internet_idx = [i for i in range(n) if SOURCE[i] == 0]
    field_idx = [i for i in range(n) if SOURCE[i] == 1]

    wsi = within_source_accuracy(cm, internet_idx)
    wsf = within_source_accuracy(cm, field_idx)
    csi = cross_source_rate(cm, internet_idx, field_idx)
    csf = cross_source_rate(cm, field_idx, internet_idx)

    return {
        "label": label,
        "macro_f1": macro_f1,
        "f1_per_class": f1_per_class,
        "within_internet_acc": wsi,
        "within_field_acc": wsf,
        "internet_to_field_rate": csi,
        "field_to_internet_rate": csf,
    }


def main() -> None:
    experiments = [
        ("exp1_ce_resnet50", "CE ResNet-50"),
        ("exp1_ce_dinov2_base", "CE DINOv2-Base"),
        ("exp2_supcon_dinov2_base", "SupCon DINOv2-Base"),
        ("exp2_supcon_convnext_tiny", "SupCon ConvNeXt-Tiny"),
        ("exp3_sphor_convnext_tiny", "SpHOR ConvNeXt-Tiny"),
        ("exp3_sphor_dinov2_base", "SpHOR DINOv2-Base"),
    ]

    results = []
    for exp_dir, label in experiments:
        r = analyze_experiment(exp_dir, label)
        if r:
            results.append(r)
            print(f"\n{'='*60}")
            print(f"  {r['label']}")
            print(f"{'='*60}")
            print(f"  Macro F1 test: {r['macro_f1']:.4f}")
            print(f"  Accuratezza within-internet:   {r['within_internet_acc']:.3f}")
            print(f"  Accuratezza within-campo:      {r['within_field_acc']:.3f}")
            print(f"  Tasso internet→campo:           {r['internet_to_field_rate']:.3f}")
            print(f"  Tasso campo→internet:           {r['field_to_internet_rate']:.3f}")
            print("  F1 per classe:")
            for i, (sp, f1) in enumerate(zip(SPECIES, r["f1_per_class"])):
                src = SRC_NAMES[SOURCE[i]]
                print(f"    {sp:30s} ({src:>8s}) F1={f1:.4f}")
            print()

    # Summary table
    print(f"\n{'='*60}")
    print("  TABELLA RIASSUNTIVA")
    print(f"{'='*60}")
    print(f"{'Modello':35s} {'Macro F1':>8s} {'W-I':>6s} {'W-F':>6s} {'I→F':>6s} {'F→I':>6s}")
    print("-" * 67)
    for r in results:
        print(
            f"{r['label']:35s} {r['macro_f1']:>8.4f} "
            f"{r['within_internet_acc']:>6.3f} {r['within_field_acc']:>6.3f} "
            f"{r['internet_to_field_rate']:>6.3f} {r['field_to_internet_rate']:>6.3f}"
        )

    output = {
        r["label"]: {
            "macro_f1": r["macro_f1"],
            "within_internet_acc": r["within_internet_acc"],
            "within_field_acc": r["within_field_acc"],
            "internet_to_field_rate": r["internet_to_field_rate"],
            "field_to_internet_rate": r["field_to_internet_rate"],
            "f1_per_class": {SPECIES[i]: r["f1_per_class"][i] for i in range(len(SPECIES))},
        }
        for r in results
    }
    out_path = Path(__file__).resolve().parent.parent / "experiments" / "domain_shift_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSalvato in {out_path}")


if __name__ == "__main__":
    main()
