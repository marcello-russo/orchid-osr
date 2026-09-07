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
    "/ 14",
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


INET = ("O. exaltata", "O. garganica", "O. incubacea", "O. sphegodes")
FLD = ("O. majellensis", "O. sphegodes_Palena")

# (exp_dir, expected_cross_errors) — structured latent space kills domain shortcut
CROSS_DOMAIN = [
    ("exp3_sphor_convnext_tiny", 0),
    ("exp3_sphor_convnext_small", 0),
    ("exp2_supcon_dinov2_base", 0),
]

# (tag, npz_a, det_a, npz_b, det_b, expected_p)
DELONG_PAIRED = [
    ("e4-knn-vs-energy", "exp4_sphor_ncd_dinov2_base", "KNN", "exp4_sphor_ncd_dinov2_base", "Energy", 0.0015),
    ("e4-knn-base-vs-tiny", "exp4_sphor_ncd_dinov2_base", "KNN", "exp4_sphor_ncd_convnext_tiny", "KNN", 0.0390),
    ("e5-knn-tiny-vs-base", "exp5_sphor_ncd_convnext_tiny", "KNN", "exp5_sphor_ncd_dinov2_base", "KNN", 0.9766),
]


def cross_errors(exp_dir: str) -> int:
    import numpy as np

    d = load_json(EXP / exp_dir / "results.json")
    tm = d["test_metrics"]
    lab = tm["confusion_matrix_labels"]
    cm = np.array(tm["confusion_matrix"])
    ii = [i for i, n in enumerate(lab) if n in INET]
    fi = [i for i, n in enumerate(lab) if n in FLD]
    return int(cm[np.ix_(ii, fi)].sum() + cm[np.ix_(fi, ii)].sum())


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
    # --- ECE (real, from softmax confidences; versioned copy) ---
    try:
        ece = float(load_json(BACKEND.parent / "documents" / "_data" / "ece.json")["ece"])
        if abs(ece - 0.058) > 0.005:
            print(f"FAIL ece: {ece} != 0.058")
            fails += 1
    except Exception as e:
        print(f"FAIL ece.json: {e}")
        fails += 1
    # --- cross-domain confusion (structured space only) ---
    for exp_dir, expected in CROSS_DOMAIN:
        try:
            got = cross_errors(exp_dir)
            if got != expected:
                print(f"FAIL cross-domain {exp_dir}: {got} != {expected}")
                fails += 1
        except Exception as e:
            print(f"FAIL cross-domain {exp_dir}: {e}")
            fails += 1
    # --- DeLong paired p-values, recomputed live ---
    try:
        import numpy as np

        from app.core.ncd.delong import delong_paired_test

        for tag, da, fa, db, fb, ep in DELONG_PAIRED:
            ya = np.load(EXP / da / "osr_scores.npz")
            yb = np.load(EXP / db / "osr_scores.npz")
            assert (ya["y_true"] == yb["y_true"]).all(), f"{tag}: y_true mismatch"
            r = delong_paired_test(ya["y_true"], ya[fa], yb[fb])
            if abs(r["p_value"] - ep) > 0.002:
                print(f"FAIL delong {tag}: p={r['p_value']:.4f} != {ep}")
                fails += 1
    except Exception as e:
        print(f"FAIL delong paired: {e}")
        fails += 1
    # --- key numbers that MUST appear in the deck text ---
    REQUIRED = [
        "0,875",
        "0,850",
        "0,839",
        "0,422",
        "0,688",
        "0,755",
        "0,843",
        "0,805",
        "0,906",
        "0,110",
        "0,83 ± 0,04",
        "0,835–0,916",
        "p=0,98",
        "0,058",
        "1.477",
    ]
    if DECK.exists():
        html = DECK.read_text(encoding="utf-8")
        for b in BANNED:
            if b in html:
                print(f"FAIL deck contains banned string: {b!r}")
                fails += 1
        for r in REQUIRED:
            if r not in html:
                print(f"FAIL deck missing required number: {r!r}")
                fails += 1
    else:
        print("WARN deck html not found, skipping banned-string check")
    n = len(CLAIMS)
    print(f"{n - fails}/{n} numeric claims OK" + (" + deck clean" if fails == 0 else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
