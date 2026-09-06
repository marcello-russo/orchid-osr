"""Defense numbers audit — single source of truth for thesis/speech/deck.

Checks every number cited in defense materials against the experiment JSONs:
- closed-set F1  -> experiments/exp{N}_*/results.json :: test_metrics.f1_macro
- OSR AUROC      -> experiments/exp{N}_sphor_ncd_*/ncd_results.json :: detectors.*.auroc
- GCD            -> experiments/exp4_sphor_ncd_*/gcd_results.json (test-pool metrics)
- banned strings -> documents/thesis_defense.html (legacy Exp9, V100S, mixed numbers)

Exit 1 on any mismatch. Intended as CI gate (F0.2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
EXP = BACKEND / "experiments"
DECK = BACKEND.parent / "documents" / "thesis_defense.html"

MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]
EXP_PREFIX = {1: "exp1_ce", 2: "exp2_supcon", 3: "exp3_sphor", 4: "exp4_sphor_ncd", 5: "exp5_sphor_ncd"}

# (claim_id, exp, model, kind, key, expected)
# kind: f1 | osr-<detector> | gcd-ss-<field> | gcd-kmx10-<field>
CLAIMS: list[tuple[str, int, str, str, str, float]] = [
    # --- Exp 1 CE test F1 ---
    ("e1", 1, "resnet18", "f1", "", 0.978),
    ("e1", 1, "resnet50", "f1", "", 0.980),
    ("e1", 1, "convnext_tiny", "f1", "", 0.821),
    ("e1", 1, "convnext_small", "f1", "", 0.793),
    ("e1", 1, "dinov2_small", "f1", "", 0.970),
    ("e1", 1, "dinov2_base", "f1", "", 0.975),
    # --- Exp 2 SupCon test F1 ---
    ("e2", 2, "resnet18", "f1", "", 0.444),
    ("e2", 2, "resnet50", "f1", "", 0.465),
    ("e2", 2, "convnext_tiny", "f1", "", 0.837),
    ("e2", 2, "convnext_small", "f1", "", 0.517),
    ("e2", 2, "dinov2_small", "f1", "", 0.888),
    ("e2", 2, "dinov2_base", "f1", "", 0.936),
    # --- Exp 3 SpHOR test F1 ---
    ("e3", 3, "resnet18", "f1", "", 0.560),
    ("e3", 3, "resnet50", "f1", "", 0.750),
    ("e3", 3, "convnext_tiny", "f1", "", 0.911),
    ("e3", 3, "convnext_small", "f1", "", 0.874),
    ("e3", 3, "dinov2_small", "f1", "", 0.894),
    ("e3", 3, "dinov2_base", "f1", "", 0.910),
    # --- Exp 4 closed F1 (5 classes) ---
    ("e4", 4, "resnet18", "f1", "", 0.590),
    ("e4", 4, "resnet50", "f1", "", 0.856),
    ("e4", 4, "convnext_tiny", "f1", "", 0.920),
    ("e4", 4, "convnext_small", "f1", "", 0.933),
    ("e4", 4, "dinov2_small", "f1", "", 0.921),
    ("e4", 4, "dinov2_base", "f1", "", 0.932),
    # --- Exp 4 OSR Ensemble AUROC ---
    ("e4osr", 4, "resnet18", "osr-Ensemble", "", 0.422),
    ("e4osr", 4, "resnet50", "osr-Ensemble", "", 0.688),
    ("e4osr", 4, "convnext_tiny", "osr-Ensemble", "", 0.850),
    ("e4osr", 4, "convnext_small", "osr-Ensemble", "", 0.839),
    ("e4osr", 4, "dinov2_small", "osr-Ensemble", "", 0.755),
    ("e4osr", 4, "dinov2_base", "osr-Ensemble", "", 0.875),
    # --- Exp 4 per-detector spot checks (speech narrative) ---
    ("e4det", 4, "dinov2_small", "osr-Energy", "", 0.540),
    ("e4det", 4, "resnet50", "osr-KNN", "", 0.776),
    ("e4det", 4, "dinov2_base", "osr-KNN", "", 0.856),
    ("e4det", 4, "dinov2_base", "osr-Energy", "", 0.745),
    # --- Exp 5 closed F1 (4 classes, test on 200) ---
    ("e5", 5, "resnet18", "f1", "", 0.808),
    ("e5", 5, "resnet50", "f1", "", 0.240),
    ("e5", 5, "convnext_tiny", "f1", "", 0.906),
    ("e5", 5, "convnext_small", "f1", "", 0.531),
    ("e5", 5, "dinov2_small", "f1", "", 0.291),
    ("e5", 5, "dinov2_base", "f1", "", 0.110),
    # --- Exp 5 OSR Ensemble AUROC ---
    ("e5osr", 5, "resnet18", "osr-Ensemble", "", 0.666),
    ("e5osr", 5, "resnet50", "osr-Ensemble", "", 0.634),
    ("e5osr", 5, "convnext_tiny", "osr-Ensemble", "", 0.843),
    ("e5osr", 5, "convnext_small", "osr-Ensemble", "", 0.718),
    ("e5osr", 5, "dinov2_small", "osr-Ensemble", "", 0.734),
    ("e5osr", 5, "dinov2_base", "osr-Ensemble", "", 0.805),
    # --- Exp 5 per-detector spot checks ---
    ("e5det", 5, "convnext_tiny", "osr-Mahalanobis", "", 0.849),
    ("e5det", 5, "dinov2_base", "osr-Mahalanobis", "", 0.835),
    ("e5det", 5, "convnext_tiny", "osr-KNN", "", 0.836),
    ("e5det", 5, "dinov2_base", "osr-KNN", "", 0.836),
    ("e5det", 5, "convnext_tiny", "osr-Energy", "", 0.703),
    # --- Exp 4 GCD: SS-KM test-pool (gcd_results.json) ---
    ("gcd4", 4, "convnext_tiny", "gcd-ss", "acc_old_test", 0.896),
    ("gcd4", 4, "dinov2_small", "gcd-ss", "acc_old_test", 0.912),
    ("gcd4", 4, "dinov2_base", "gcd-ss", "acc_old_test", 0.932),
    ("gcd4", 4, "dinov2_base", "gcd-ss", "acc_new_test", 0.02),
    ("gcd4", 4, "resnet50", "gcd-ss", "acc_new_test", 0.44),
    ("gcd4", 4, "dinov2_base", "gcd-ss", "nmi_pool", 0.928),
    # --- Exp 4 GCD: KMeans x10 seeds ---
    ("gcd4k", 4, "dinov2_base", "gcd-kmx10", "acc_new_mean", 0.826),
    ("gcd4k", 4, "dinov2_base", "gcd-kmx10", "acc_new_std", 0.035),
    ("gcd4k", 4, "dinov2_base", "gcd-kmx10", "acc_all_mean", 0.873),
]

# Strings that must NOT appear in the rebuilt deck
BANNED = [
    "Exp 9",
    "exp9",
    "V100S",
    "5120 CUDA",
    "650 GPU-ore",
    "Senza LoRA",
    "senza compromessi",
    "Exp 1–5 affetti",
    "Exp 1-5 affetti",
]

TOL = 0.002


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def actual_value(exp: int, model: str, kind: str, key: str) -> float:
    if kind == "f1":
        d = load_json(EXP / f"{EXP_PREFIX[exp]}_{model}" / "results.json")
        return float(d["test_metrics"]["f1_macro"])
    if kind.startswith("osr-"):
        det = kind.split("-", 1)[1]
        d = load_json(EXP / f"{EXP_PREFIX[exp]}_{model}" / "ncd_results.json")
        return float(d["detectors"][det]["auroc"])
    if kind == "gcd-ss":
        d = load_json(EXP / f"{EXP_PREFIX[exp]}_{model}" / "gcd_results.json")
        return float(d["ss_kmeans"][key])
    if kind == "gcd-kmx10":
        d = load_json(EXP / f"{EXP_PREFIX[exp]}_{model}" / "gcd_results.json")
        return float(d["kmeans"][key])
    raise ValueError(kind)


def main() -> int:
    fails = 0
    for cid, exp, model, kind, key, expected in CLAIMS:
        try:
            got = actual_value(exp, model, kind, key)
        except Exception as e:
            print(f"FAIL {cid} exp{exp}/{model}/{kind}/{key}: {e}")
            fails += 1
            continue
        ok = abs(got - expected) <= TOL
        if not ok:
            print(f"FAIL {cid} exp{exp}/{model}/{kind}/{key}: json={got:.4f} claimed={expected}")
            fails += 1
    if DECK.exists():
        html = DECK.read_text(encoding="utf-8")
        for b in BANNED:
            if b in html:
                print(f"FAIL deck contains banned string: {b!r}")
                fails += 1
    else:
        print("WARN deck html not found, skipping banned-string check")
    n = len(CLAIMS)
    print(f"{n - fails}/{n} numeric claims OK" + (" + deck clean" if fails == 0 else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
