#!/usr/bin/env python3
"""Ablazione SpHOR su una fold di Studio II-A (DINOv2-Small).

Quattro varianti, stesso protocollo, stessa fold, un solo modello:

    A  SupCon puro                         (baseline I-B)
    B  + vMF k=10, repulsione squared 0.01 (quello che c'era)
    C  + vMF k=10, repulsione hinge 0.1    (soglia coseno 0.3)
    D  + vMF k=25, repulsione hinge 0.1

La metrica che decide e' l'AUROC open-set (noto vs O. garganica), non l'F1
closed-set: e' li' che SpHOR dovrebbe pagare. I rivelatori sono quelli della
tesi (Mahalanobis, KNN k=5, Energy, Ensemble), addestrati sugli embedding del
train noto e valutati sul test completo.

Uso:
    scripts/ablation_sphor.py                 # addestra le 4 varianti e valuta
    scripts/ablation_sphor.py --eval-only     # valuta le dir gia' presenti
    scripts/ablation_sphor.py --only A,C

Output: `ablation_sphor.json` + tabella a schermo.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.cli.definitions import TRAIN_CLASSES  # noqa: E402
from app.core.models import create_model, create_transform  # noqa: E402
from app.core.ncd.detector import get_detector  # noqa: E402
from app.preprocessing.dataset_tool import ImageFolderWithName  # noqa: E402

MODEL = "dinov2_small"
EXP = 4  # II-A: O. garganica nascosta
FOLD = 0
NOVEL = "O. garganica"

VARIANTS: dict[str, dict] = {
    "A": {"use_prototypes": False},
    "B": {"use_prototypes": True, "vmf_kappa": 10.0, "repulse_mode": "squared", "repulse_weight": 0.01},
    "C": {
        "use_prototypes": True,
        "vmf_kappa": 10.0,
        "repulse_mode": "hinge",
        "repulse_weight": 0.1,
        "repulse_threshold": 0.3,
    },
    "D": {
        "use_prototypes": True,
        "vmf_kappa": 25.0,
        "repulse_mode": "hinge",
        "repulse_weight": 0.1,
        "repulse_threshold": 0.3,
    },
}


def out_dir(name: str) -> Path:
    return BACKEND / "experiments" / f"exp4_sphor_ncd_{MODEL}_abl-{name.lower()}"


def map_labels(labels: np.ndarray, dataset_classes: list[str], model_classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Rimappa le label del dataset sugli indici del modello e scarta il resto."""
    mapping = {i: model_classes.index(c) for i, c in enumerate(dataset_classes) if c in model_classes}
    keep = np.isin(labels, list(mapping))
    mapped = np.array([mapping[int(label)] for label in labels[keep]], dtype=np.int64)
    return mapped, keep


def extract(model, loader, device):
    model.eval()
    embs, logits, labels = [], [], []
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            embs.append(model.get_embeddings(inputs).cpu().numpy())
            logits.append(model(inputs).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(embs), np.concatenate(logits), np.concatenate(labels)


def evaluate(name: str) -> dict:
    checkpoint = out_dir(name) / f"best_fold_{FOLD}.pt"
    if not checkpoint.exists():
        return {"error": f"manca {checkpoint}"}

    model_classes = [c for c in TRAIN_CLASSES if c != NOVEL]
    entry = create_model(MODEL, num_classes=len(model_classes), pretrained=False, normalize_embeddings=True)
    entry.load_weights(str(checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry = entry.to(device)

    size = "224x224"
    transform = create_transform(MODEL)
    train_ds = ImageFolderWithName(str(BACKEND / "dataset" / f"cropped_{size}" / "train"), transform=transform)
    test_ds = ImageFolderWithName(str(BACKEND / "dataset" / f"cropped_{size}" / "test"), transform=transform)

    tr_emb, _, tr_lab = extract(entry, DataLoader(train_ds, batch_size=32, shuffle=False), device)
    te_emb, te_logits, te_lab = extract(entry, DataLoader(test_ds, batch_size=32, shuffle=False), device)

    # Il train si filtra alle classi note (i prototipi/centroidi sono solo per
    # quelle); il test NO: la specie nascosta e' il positivo dell'AUROC, e
    # filtrarla via prima di calcolare `is_novel` lasciava zero positivi
    # (AUROC = nan, visto nella prima passata).
    tr_mapped, keep_tr = map_labels(tr_lab, train_ds.classes, model_classes)
    novel_idx = test_ds.classes.index(NOVEL)
    is_novel = (te_lab == novel_idx).astype(int)
    if is_novel.sum() == 0 or is_novel.sum() == len(is_novel):
        raise RuntimeError(f"AUROC impossibile: {is_novel.sum()} positivi su {len(is_novel)} nel test")

    results = {
        "model": MODEL,
        "fold": FOLD,
        "novel": NOVEL,
        "n_train": int(keep_tr.sum()),
        "n_test": int(len(te_lab)),
        "n_novel": int(is_novel.sum()),
    }
    for detector_name in ("Mahalanobis", "KNN", "Energy", "Ensemble"):
        det = get_detector(detector_name)
        det.fit(tr_emb[keep_tr], tr_mapped)
        scores = det.score(te_emb, logits=te_logits)
        auroc = float(roc_auc_score(is_novel, scores))
        fpr, tpr, _ = roc_curve(is_novel, scores)
        tpr_at_fpr95 = float(tpr[np.searchsorted(fpr, 0.05, side="left")])
        results[detector_name] = {"auroc": round(auroc, 4), "tpr_at_fpr95": round(tpr_at_fpr95, 4)}
    return results


def checkpoint_hashes(names: list[str]) -> dict[str, str | None]:
    """md5 dei checkpoint: due varianti con lo stesso file sono un errore, non un risultato."""
    import hashlib

    out: dict[str, str | None] = {}
    for name in names:
        path = out_dir(name) / f"best_fold_{FOLD}.pt"
        out[name] = hashlib.md5(path.read_bytes()).hexdigest()[:12] if path.exists() else None
    return out


def train(name: str) -> None:
    from app.cli.runner import run_experiment

    kwargs = dict(VARIANTS[name])
    start = time.time()
    print(f"=== variante {name}: {kwargs} ===", flush=True)
    # Budget fisso per tutte e quattro: con early stopping la variante A
    # (senza prototipi) si ferma molto prima, e il confronto mischierebbe
    # l'effetto dei termini SpHOR con un budget di addestramento diverso.
    run_experiment(
        exp_id=EXP,
        model_variant=MODEL,
        fold_index=FOLD,
        epochs=60,
        ce_epochs=50,
        patience=0,
        # Il modello da valutare e' quello con cui il metodo finisce (ultima
        # epoca, peso pieno), non l'epoca migliore per KNN: su SpHOR quella
        # cade nel warm-up e due varianti darebbero lo stesso checkpoint.
        restore_best=False,
        tag=f"abl-{name.lower()}",
        warmup_epochs=20,
        ramp_epochs=20,
        **kwargs,
    )
    print(f"=== {name} finita in {(time.time() - start) / 3600:.2f} h ===", flush=True)


COMPLEXITY = ("A", "B", "C", "D")
MARGIN = 0.01  # quanto deve battere SupCon per giustificare il metodo
TIE = 0.005  # a parita' quasi, si sceglie la variante piu' semplice


def decide(report: dict) -> dict:
    """Regola fissata prima di vedere i numeri.

    Vincente = la variante piu' semplice entro TIE dal miglior AUROC Ensemble.
    Se non batte SupCon di almeno MARGIN, vince A: il metodo non paga e non si
    lancia nessun grid.
    """
    auroc = {n: (report.get(n) or {}).get("Ensemble", {}).get("auroc") for n in COMPLEXITY}
    auroc = {n: v for n, v in auroc.items() if isinstance(v, (int, float))}
    if "A" not in auroc or len(auroc) < 2:
        return {"winner": None, "reason": "valutazione incompleta"}
    best = max(auroc.values())
    simplest = next(n for n in COMPLEXITY if auroc.get(n) is not None and auroc[n] >= best - TIE)
    margin = auroc[simplest] - auroc["A"]
    if margin < MARGIN:
        return {
            "winner": "A",
            "auroc": round(auroc["A"], 4),
            "margin": round(margin, 4),
            "reason": f"nessuna variante SpHOR batte SupCon di almeno {MARGIN} AUROC "
            f"(migliore: {simplest} {auroc[simplest]:.4f})",
            "run_grid": False,
        }
    return {
        "winner": simplest,
        "auroc": round(auroc[simplest], 4),
        "margin": round(margin, 4),
        "reason": f"{simplest} batte SupCon di {margin:+.4f} AUROC (Ensemble, fold 0)",
        "run_grid": True,
        "params": VARIANTS[simplest],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--only", default="", help="Sottoinsieme di varianti, es. A,C")
    parser.add_argument("--decide", action="store_true", help="Legge ablation_sphor.json e scrive il verdetto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(__file__).resolve().parents[1]
    if args.decide:
        report = json.loads((root / "ablation_sphor.json").read_text(encoding="utf-8"))
        verdict = decide(report)
        (root / "ablation_decision.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(verdict, indent=2, ensure_ascii=False))
        return 0
    names = [n.strip().upper() for n in args.only.split(",") if n.strip()] or list(VARIANTS)

    if not args.eval_only:
        for name in names:
            if (out_dir(name) / f"fold_{FOLD}.json").exists():
                print(f"=== {name} gia' addestrata, salto ===", flush=True)
                continue
            train(name)

    hashes = checkpoint_hashes(names)
    print("checkpoint:", hashes, flush=True)
    identical = [
        (a, b) for i, a in enumerate(names) for b in names[i + 1 :] if hashes[a] and hashes[a] == hashes[b]
    ]
    if identical:
        print("!!! ATTENZIONE: varianti con lo STESSO checkpoint:", identical, flush=True)
        print("!!! I numeri non sono un confronto tra metodi: controlla selezione e schedule.", flush=True)

    report = {"_checkpoints": hashes, "_identical": identical}
    for name in names:
        report[name] = evaluate(name)
        print(f"--- {name}: {json.dumps(report[name], ensure_ascii=False)}", flush=True)

    out = Path(__file__).resolve().parents[1] / "ablation_sphor.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nscritto {out}", flush=True)

    print("\nAUROC (noto vs garganica), fold 0, DINOv2-Small")
    print(f"{'variante':8s} " + " ".join(f"{d:>12s}" for d in ("Mahalanobis", "KNN", "Energy", "Ensemble")))
    for name in names:
        row = report.get(name, {})
        cells = []
        for det in ("Mahalanobis", "KNN", "Energy", "Ensemble"):
            value = (row.get(det) or {}).get("auroc")
            cells.append(f"{value:>12.4f}" if value is not None else f"{'-':>12s}")
        print(f"{name:8s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
