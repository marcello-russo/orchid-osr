# app/api/serializers/experiments.py
"""Lettura degli esperimenti da disco e costruzione dei payload delle route.

Le route fanno I/O e validazione degli input; qui vivono le regole di dominio
(quale F1 è quello ufficiale, come si compongono i fold, le ricette per numero
di esperimento), così restano testabili senza passare da un client HTTP.
"""

import json
from pathlib import Path
from typing import Any

from app.api.serializers.folds import normalize_fold

RECIPES: dict[int, dict[str, Any]] = {
    1: {
        "phases": ["Cross-Entropy (fase singola)"],
        "description": "Baseline supervisionata: cross-entropia su 6 specie, "
        "backbone e testa allenati insieme sul clean split.",
        "recipe": "Training monofase con loss Cross-Entropy standard su 6 classi. "
        "L'intera rete (backbone + testa) è addestrata insieme. "
        "150 epoche, early stopping (patience 15), Adam (lr=0.001), batch 8. "
        "5-fold stratificata, seed 57, clean split 2.192/300. "
        "Scopo: baseline per confronto con i paradigmi contrastivi.",
    },
    2: {
        "phases": ["Fase 1: SupCon (pretraining contrastivo)", "Fase 2: CE fine-tuning (linear probe)"],
        "description": "Pretraining contrastivo + testa lineare " "su backbone congelato, 6 classi.",
        "recipe": "Training a due fasi per embedding strutturati. "
        "Fase 1: SupConLoss (τ=0.07, proiezione a 128 dim), validazione KNN. "
        "Fase 2: backbone congelato, sola testa lineare con CrossEntropy. "
        "Stesso split, seed e protocollo di Exp 1.",
    },
    3: {
        "phases": ["Fase 1: SupCon + SpHOR", "Fase 2: CE fine-tuning"],
        "description": "Come Exp 2 con prototipi ortogonali sferici (SpHOR) in Fase 1, su 6 classi note.",
        "recipe": "Come Exp 2 ma con Spherical Orthogonal Prototypes: prototipi L2-normalizzati "
        "sull'ipersfera con loss di repulsione ortogonale (λ=0.01). "
        "Fase 2 identica a Exp 2. Valutazione closed-set, nessuna novel class.",
    },
    4: {
        "phases": ["Fase 1: SupCon + SpHOR", "Fase 2: CE fine-tuning"],
        "description": "Come Exp 3 ma O. garganica esclusa dal training " "(5 note): novel nascosta per OSR e GCD.",
        "recipe": "Strutturalmente identico a Exp 3 (stessa loss, stessi iperparametri). "
        "L'unica differenza: O. garganica è esclusa dal training (5 classi note) "
        "e riservata al test come novel class nascosta per OSR (4 rivelatori) e GCD.",
    },
    5: {
        "phases": ["Fase 1: SupCon + SpHOR", "Fase 2: CE fine-tuning"],
        "description": "Come Exp 4 con due novel nascoste " "(garganica + majellensis, 4 note).",
        "recipe": "Identico a Exp 4 ma con due novel nascoste (garganica + majellensis). "
        "Stessi 4 rivelatori OSR: il rilevamento scala con due novel?",
    },
    6: {
        "phases": ["Cross-Entropy (fase singola)"],
        "recipe": "Training monofase con loss Cross-Entropy standard. "
        "L'intera rete (backbone + testa classificatore) è addestrata insieme. "
        "Ottimizzatore: Adam (lr=0.001). Epoche: 150 (early stopping patience=15). "
        "Batch: 8 (effettivo 32 con accumulazione gradiente). "
        "5-fold stratified cross-validation. "
        "Scopo: baseline supervisionata per confronto con metodi contrastivi.",
    },
    7: {
        "phases": ["Fase 1: SupCon (pretraining contrastivo)", "Fase 2: CE fine-tuning (linear probe)"],
        "recipe": "Training a due fasi per embedding strutturati.\n\n"
        "Fase 1 — SupCon (Supervised Contrastive Learning):\n"
        "  • Loss: SupConLoss — attrae embedding della stessa specie, "
        "respinge specie diverse sull'ipersfera.\n"
        "  • Temperatura τ = 0.07 (valore standard da [8]).\n"
        "  • Proiezione: singolo strato lineare (embedding_dim → 128), scartato dopo Fase 1.\n"
        "  • Backbone + projection head addestrati insieme.\n"
        "  • Metrica di validazione: KNN accuracy (k=1).\n"
        "  • Ottimizzatore: AdamW per-architettura "
        "(ResNet: lr=0.001, ConvNeXt: lr=0.0001, DINOv2: lr=0.0002).\n\n"
        "Fase 2 — CE Fine-Tuning (Linear Probe):\n"
        "  • Backbone congelato (preserva struttura contrastiva).\n"
        "  • Solo testa lineare addestrata con Cross-Entropy.\n"
        "  • LR più alto (0.01) perché solo la testa casuale si addestra.\n"
        "  • Adam, max 50 epoche, early stopping (patience=15).",
    },
    8: {
        "phases": ["Fase 1: SupCon + SpHOR", "Fase 2: CE fine-tuning"],
        "recipe": "Stessa struttura a due fasi di Exp 2, con regolarizzazione aggiuntiva in Fase 1.\n\n"
        "Fase 1 — SupCon + SpHOR (Prototipi Ortogonali Sferici):\n"
        "  • Warm-up (20 ep): sola SupCon, nessun vincolo geometrico.\n"
        "  • Poi un prototipo per classe (dim 128, L2-normalizzato) inizializzato "
        "sul centroide di classe delle feature proiettate.\n"
        "  • Loss: SupCon + w(e)·vMF + λ_rep·L_rep, con w(e) lineare 0→1 in 20 ep.\n"
        "  • vMF hard assignment (κ=10): ogni foto è tirata verso il prototipo "
        "della propria classe, così i prototipi seguono i dati.\n"
        "  • L_rep = (1/K(K-1)) Σ_{i≠j} cos(θ_ij)² (λ_rep=0.01) — tiene i prototipi "
        "ortogonali e massimizza la copertura angolare.\n"
        "  • I prototipi stanno nell'optimizer e vengono salvati in un sidecar "
        "(.sphor.pt) accanto al checkpoint.\n"
        "  • Stessi ottimizzatori e scheduling di Exp 2.\n\n"
        "Fase 2 — CE Fine-Tuning:\n"
        "  • Identica a Exp 2: backbone congelato, testa lineare.\n\n"
        "Valutazione: closed-set su 6 classi note, nessuna novel class.",
    },
    9: {
        "phases": ["Fase 1: SupCon + SpHOR", "Fase 2: CE fine-tuning"],
        "recipe": "Strutturalmente identico a Exp 3 (stessa loss, stessi iperparametri).\n"
        "L'unica differenza: O. garganica è esclusa dal training (5 classi note) "
        "e riservata al test come novel class nascosta per la valutazione OSR/NCD.",
    },
    10: {
        "phases": ["Fase 1: MGH-SupCon (Multi-Granularità Hard-negative)", "Fase 2: CE fine-tuning"],
        "recipe": "Training a due fasi potenziato con due innovazioni chiave rispetto a Exp 3.\n\n"
        "Fase 1 — MGH-SupCon:\n"
        "  • Hard Negative Mining: pesi per coppia di classi:\n"
        "    — (O. majellensis, O. sphegodes): 5× (coppia più confusa).\n"
        "    — (O. sphegodes, O. sphegodes_Palena): 3×.\n"
        "    — Altre coppie: 1× (standard).\n"
        "  • Features Multi-Granularità (MG):\n"
        "    — Modelli ViT (DINOv2): concatenano token CLS (globale) + "
        "patch token con attention pooling (locale) + CLS intermedio (regionale).\n"
        "    — Modelli CNN (ResNet, ConvNeXt): concatenano features "
        "del penultimo e ultimo stadio dopo global average pooling.\n"
        "  • Dimensione embedding combinato: 1152 (DINOv2) o 3072 (ResNet50).\n"
        "  • Embedding multi-granularità nella SupCon loss con la stessa proiezione lineare singola.\n"
        "  • Stessi ottimizzatori e scheduling di Exp 3.\n\n"
        "Fase 2 — CE Fine-Tuning:\n"
        "  • Backbone congelato, testa lineare (come Exp 2-9).\n\n"
        "Valutazione: O. sphegodes_Napoli novel class (solo in test). "
        "Obiettivo: miglior rilevamento novel via embedding più separati.",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data: Any = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: atteso un oggetto JSON")
    return data


def read_ncd_metrics(d: Path) -> dict[str, Any]:
    """Legge metriche NCD da ncd_results.json, se presente."""
    ncd_path = d / "ncd_results.json"
    if not ncd_path.exists():
        return {}
    try:
        ncd_data = json.loads(ncd_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

    metrics: dict[str, Any] = {}
    detectors = ncd_data.get("detectors", {})
    if isinstance(detectors, dict):
        # Regola ufficiale (tesi): Ensemble, stesso rivelatore per tutti.
        # Solo se Ensemble manca si ripiega sul migliore, dichiarandolo.
        ens = detectors.get("Ensemble")
        if isinstance(ens, dict) and isinstance(ens.get("auroc"), (int, float)):
            metrics["ncd_auroc"] = ens["auroc"]
            metrics["ncd_detector"] = "Ensemble"
        else:
            best_auroc = -1.0
            best_detector = None
            for name, det_data in detectors.items():
                if not isinstance(det_data, dict):
                    continue
                auroc = det_data.get("auroc")
                if isinstance(auroc, (int, float)) and auroc > best_auroc:
                    best_auroc = auroc
                    best_detector = name
            if best_auroc >= 0:
                metrics["ncd_auroc"] = best_auroc
                metrics["ncd_detector"] = best_detector

    clustering = ncd_data.get("clustering", {})
    if isinstance(clustering, dict):
        best_nmi = -1.0
        for name, cl_data in clustering.items():
            if not isinstance(cl_data, dict):
                continue
            nmi = cl_data.get("nmi")
            if isinstance(nmi, (int, float)) and nmi > best_nmi:
                best_nmi = nmi
        if best_nmi >= 0:
            metrics["ncd_clustering_nmi"] = best_nmi

    return metrics


def perfold_stats(exp_dir: Path) -> tuple[float | None, float | None]:
    """(mean, std) F1 test da perfold_test.json, o (None, None) se manca."""
    from app.core.perfold import load_perfold

    pf = load_perfold(exp_dir)
    if not pf:
        return None, None
    try:
        return float(pf["f1_mean"]), float(pf["f1_std"])
    except (ValueError, KeyError):
        return None, None


def override_test_metrics(exp_dir: Path, results: dict[str, Any] | None) -> None:
    """Sostituisce test_metrics con la confusion del best fold sul test ufficiale."""
    if not results:
        return
    from app.core.perfold import load_perfold

    pf = load_perfold(exp_dir)
    if not pf or not pf.get("best_confusion_matrix"):
        return
    try:
        import numpy as np

        cm = np.array(pf["best_confusion_matrix"], dtype=float)
        labels = list(pf.get("best_confusion_matrix_labels") or [])
        total = cm.sum()
        if total <= 0 or cm.shape[0] != len(labels):
            return
        acc = float(np.trace(cm) / total)
        per_class: dict[str, dict[str, float]] = {}
        f1s = []
        for i, name in enumerate(labels):
            tp = cm[i, i]
            pred_pos = cm[:, i].sum()
            actual = cm[i, :].sum()
            prec = float(tp / pred_pos) if pred_pos > 0 else 0.0
            rec = float(tp / actual) if actual > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
            per_class[name] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": int(actual),
            }
        results["test_metrics"] = {
            "accuracy": acc,
            "f1_macro": float(sum(f1s) / len(f1s)) if f1s else 0.0,
            "precision_macro": None,
            "recall_macro": None,
            "confusion_matrix": cm.astype(int).tolist(),
            "confusion_matrix_labels": labels,
            "per_class_metrics": per_class,
        }
    except (ValueError, KeyError):
        return


def build_experiment_response(
    exp_id: str,
    exp_dir: Path,
    config: dict[str, Any],
    folds: list[dict[str, Any]],
    results: dict[str, Any] | None = None,
    exp_data: dict[str, Any] | None = None,
    last_modified: str | None = None,
    ncd_full: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Costruisce risposta standard per un esperimento."""
    # Normalize results from DB row or JSON dict
    if exp_data:
        results = {
            "mean_f1": exp_data.get("mean_f1"),
            "std_f1": exp_data.get("std_f1"),
            "mean_accuracy": exp_data.get("mean_accuracy"),
            "test_metrics": {
                "f1_macro": exp_data.get("test_f1"),
                "accuracy": exp_data.get("test_accuracy"),
            }
            if exp_data.get("test_f1") is not None
            else None,
            "best_fold": exp_data.get("best_fold"),
            "ncd_auroc": exp_data.get("ncd_auroc"),
            "ncd_aupr": exp_data.get("ncd_aupr"),
            "ncd_detector": exp_data.get("ncd_detector"),
            "ncd_clustering_nmi": exp_data.get("ncd_clustering_nmi"),
        }

    is_ncd = bool(config.get("ncd_mode")) or bool((exp_data or {}).get("is_ncd"))

    # Come nella lista: F1 test ufficiale (medie ± std 5 fold) se disponibile.
    pf_mean, pf_std = perfold_stats(exp_dir)
    if pf_mean is None:
        pf_mean = (
            ((results or {}).get("test_metrics") or {}).get("f1_macro")
            if ((results or {}).get("test_metrics") or {}).get("f1_macro") is not None
            else (results or {}).get("mean_f1")
        )
    if pf_std is None:
        pf_std = (results or {}).get("std_f1")

    # Normalize all folds (handles SupCon two-phase → flat + phase_epochs)
    normalized_folds = []
    for f in folds:
        if isinstance(f, dict):
            normalized_folds.append(normalize_fold(f))
        else:
            normalized_folds.append(f)

    return {
        "id": exp_id,
        "config": config,
        "folds": normalized_folds,
        "results": results,
        "last_modified": last_modified,
        "detectors": (ncd_full or {}).get("detectors"),
        "openset": (ncd_full or {}).get("openset"),
        "clustering": (ncd_full or {}).get("clustering"),
        "summary": {
            "model_variant": config.get("model_variant") or (results or {}).get("model_variant"),
            "type": "kfold" if folds else "baseline",
            "num_folds": len(folds),
            "mean_accuracy": (results or {}).get("mean_accuracy"),
            "std_accuracy": (results or {}).get("std_accuracy"),
            "mean_f1": pf_mean,
            "std_f1": pf_std,
            "test_accuracy": ((results or {}).get("test_metrics") or {}).get("accuracy"),
            "test_f1": ((results or {}).get("test_metrics") or {}).get("f1_macro"),
            "is_ncd": is_ncd,
            "ncd_auroc": (results or {}).get("ncd_auroc"),
            "ncd_aupr": (results or {}).get("ncd_aupr"),
            "ncd_detector": (results or {}).get("ncd_detector"),
            "ncd_clustering_nmi": (results or {}).get("ncd_clustering_nmi"),
        },
    }
