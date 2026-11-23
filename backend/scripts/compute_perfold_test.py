"""Valuta i checkpoint best_fold_0..4 di ogni esperimento sul test set.

Replica esattamente la logica di test del runner (stessi test_dir, class_names,
num_classes) e salva perfold_test.json con F1/acc per fold + mean/std.
Serve per riportare F1 test come media ± std sui 5 fold nel deck.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

import torch  # noqa: E402

from app.cli.definitions import ALL_CLASSES, EXPERIMENTS, PROJECT_ROOT  # noqa: E402
from app.core.kfold.testing import final_test  # noqa: E402
from app.core.models import get_registry_entry  # noqa: E402

logging.basicConfig(level=logging.WARNING)

PREFIX_TO_EXP = {
    "exp1_ce_": 1,
    "exp2_supcon_": 2,
    "exp3_sphor_": 3,
    "exp4_sphor_ncd_": 4,
    "exp5_sphor_ncd_": 4,  # placeholder, fixed below
}


def resolve_exp(dirname: str) -> int | None:
    for prefix, num in [
        ("exp1_ce_", 1),
        ("exp2_supcon_", 2),
        ("exp3_sphor_", 3),
        ("exp4_sphor_ncd_", 4),
        ("exp5_sphor_ncd_", 5),
    ]:
        if dirname.startswith(prefix):
            return num
    return None


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    exp_root = BACKEND / "experiments"
    summary = {}
    for dirname in sorted(os.listdir(exp_root)):
        exp_num = resolve_exp(dirname)
        if exp_num is None:
            continue
        out_dir = exp_root / dirname
        res_path = out_dir / "results.json"
        if not res_path.exists():
            continue
        results = json.loads(res_path.read_text())
        model_variant = results.get("model_variant")
        if not model_variant:
            continue
        pts = [out_dir / f"best_fold_{i}.pt" for i in range(5)]
        if not all(p.exists() for p in pts):
            print(f"SKIP {dirname}: checkpoint mancanti", flush=True)
            continue
        definition = EXPERIMENTS[exp_num]
        exclude = set(definition.get("exclude_classes") or [])
        class_names = [c for c in definition.get("test_classes", ALL_CLASSES) if c not in exclude]
        num_classes = definition.get("num_classes", len(class_names))
        _w, _h = get_registry_entry(model_variant).input_size
        # `final_test` scarta i campioni di classi che il modello non ha, quindi
        # per II-A/II-B basta il test completo: il F1 resta sulle sole classi
        # note. Prima qui si faceva SKIP se mancava `test_ncd_exp4`, una cartella
        # creata a mano e assente dal dataset HF: su una macchina pulita il
        # per-fold test di II-A/II-B non era riproducibile.
        # Per gli esperimenti senza esclusioni resta il 256x512 di sempre: il
        # resize letterbox da 224 a 224 non e' identico a quello da 256x512, e
        # cambiare la sorgente cambierebbe i numeri della tesi.
        size_dir = f"cropped_{_w}x{_h}" if exclude else "cropped_256x512"
        test_dir = str(PROJECT_ROOT / "dataset" / size_dir / "test")
        if not os.path.exists(test_dir):
            print(f"SKIP {dirname}: manca {test_dir}", flush=True)
            continue

        f1s, accs = [], []
        per_class_f1 = {}
        cms = {}
        for i, pt in enumerate(pts):
            tm = final_test(str(pt), model_variant, num_classes, test_dir, class_names, 32, device)
            f1s.append(float(tm["f1_macro"]))
            accs.append(float(tm["accuracy"]))
            for cls, vals in (tm.get("f1_per_class") or {}).items():
                per_class_f1.setdefault(cls, []).append(
                    float(vals) if isinstance(vals, (int, float)) else float(vals.get("f1", 0))
                )
            cms[i] = {
                "confusion_matrix": tm.get("confusion_matrix"),
                "confusion_matrix_labels": tm.get("confusion_matrix_labels"),
            }
        # sanity: il best fold deve riprodurre test_metrics entro tolleranza
        best_fold = results.get("best_fold", 0)
        stored = (results.get("test_metrics") or {}).get("f1_macro")
        if stored is not None and abs(f1s[best_fold] - stored) > 0.005:
            print(
                f"WARN {dirname}: fold{best_fold} F1={f1s[best_fold]:.4f} != stored {stored:.4f} (test set cambiato)",
                flush=True,
            )
        new_best = int(np.argmax(f1s))
        best_cm = cms[new_best]
        # cross-domain errors: INET (exaltata/garganica/incubacea/sphegodes) vs FLD (majellensis/palena)
        lab = best_cm.get("confusion_matrix_labels") or []
        cm = np.array(best_cm.get("confusion_matrix") or [])
        INET = ("O. exaltata", "O. garganica", "O. incubacea", "O. sphegodes")
        FLD = ("O. majellensis", "O. sphegodes_Palena")
        cross_errors = None
        exa_sph = None
        if cm.size and len(lab):
            ii = [i for i, n in enumerate(lab) if n in INET]
            fi = [i for i, n in enumerate(lab) if n in FLD]
            cross_errors = int(cm[np.ix_(ii, fi)].sum() + cm[np.ix_(fi, ii)].sum())
            if "O. exaltata" in lab and "O. sphegodes" in lab:
                exa_sph = int(cm[lab.index("O. exaltata"), lab.index("O. sphegodes")])
        payload = {
            "model_variant": model_variant,
            "test_dir": test_dir,
            "class_names": class_names,
            "f1_per_fold": [round(v, 6) for v in f1s],
            "acc_per_fold": [round(v, 6) for v in accs],
            "f1_mean": round(float(np.mean(f1s)), 6),
            "f1_std": round(float(np.std(f1s)), 6),
            "acc_mean": round(float(np.mean(accs)), 6),
            "acc_std": round(float(np.std(accs)), 6),
            "f1_per_class_mean": {k: round(float(np.mean(v)), 6) for k, v in per_class_f1.items()},
            "f1_per_class_std": {k: round(float(np.std(v)), 6) for k, v in per_class_f1.items()},
            "best_new_fold": new_best,
            "best_confusion_matrix": best_cm.get("confusion_matrix"),
            "best_confusion_matrix_labels": best_cm.get("confusion_matrix_labels"),
            "cross_errors_new": cross_errors,
            "exa_to_sph_new": exa_sph,
            "stored_test_f1": stored,
        }
        (out_dir / "perfold_test.json").write_text(json.dumps(payload, indent=2))
        summary[dirname] = payload
        print(
            f"DONE {dirname}: F1 folds={[round(v,4) for v in f1s]} "
            f"mean={payload['f1_mean']:.4f} std={payload['f1_std']:.4f} (stored {stored})",
            flush=True,
        )
    print("FINITO", flush=True)


if __name__ == "__main__":
    main()
