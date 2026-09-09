# backend/app/api/routes/experiments.py
import datetime
import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import ExperimentDetailResponse, ExperimentSummary, FoldDataResponse
from app.api.weights import _EXP_META
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _get_experiments_dir() -> Path:
    return Path(settings.results_dir)


def _validate_id(value: str) -> str:
    """Sanitize path parameter to prevent path traversal."""
    if ".." in value or "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail="Invalid identifier")
    return value


def _parse_exp_num(name: str) -> int:
    """Extract experiment number from directory name for sorting.
    'exp10_mgh_ncd_resnet18' → 10, 'exp1_ce' → 6.
    """
    m = re.search(r"exp(\d+)", name)
    return int(m.group(1)) if m else 999


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_ncd_metrics(d: Path) -> dict:
    """Legge metriche NCD da ncd_results.json, se presente."""
    ncd_path = d / "ncd_results.json"
    if not ncd_path.exists():
        return {}
    try:
        ncd_data = json.loads(ncd_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

    metrics: dict = {}
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


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments():
    """List all experiments with summary metrics."""
    exp_dir = _get_experiments_dir()
    if not exp_dir.exists():
        return []

    experiments = []
    for d in sorted(exp_dir.iterdir(), key=lambda p: _parse_exp_num(p.name)):
        if not d.is_dir():
            continue

        # Allowlist prefissi correnti: taglia fuori dir legacy appiattite dal
        # sync HF (exp6_clean_split_*, exp9_ncd_*, ...) e cartelle di lavoro
        # (_archived, _backup_*, ...). Exp 10 resta escluso: non riaddestrato
        # con le correzioni metodologiche, risultati confounded e rimossi.
        if not any(d.name.startswith(p) for p in _EXP_META):
            continue

        config_path = d / "config.json"
        results_path = d / "results.json"

        # Need at least config.json or results.json
        if not config_path.exists() and not results_path.exists():
            continue

        config = _load_json(config_path) if config_path.exists() else {}
        results = _load_json(results_path) if results_path.exists() else None
        # stessi test_metrics del detail (best fold sul test ufficiale)
        _override_test_metrics(d, results)

        fold_files = sorted(d.glob("fold_*.json"))
        has_folds = len(fold_files) > 0
        config_type = (config or {}).get("type", "")
        exp_type = "kfold" if has_folds else ("kfold" if config_type == "kfold" else "baseline")

        # Numero di fold: file individuali o fold_results in results.json
        if has_folds:
            num_folds = len(fold_files)
            folds_done = len(fold_files)
        elif results and results.get("fold_results"):
            num_folds = len(results["fold_results"])
            folds_done = len(results["fold_results"])
        else:
            num_folds = (config or {}).get("num_folds", 5)
            folds_done = 0

        has_ncd_file = (d / "ncd_results.json").exists()
        is_ncd = bool(config.get("ncd_mode"))
        ncd_metrics = _read_ncd_metrics(d) if has_ncd_file else {}

        test_accuracy = results.get("test_metrics", {}).get("accuracy") if results else None

        # F1 test dal test ufficiale: medie ± std sui 5 fold (perfold_test.json).
        # Fallback al vecchio comportamento solo se il file manca.
        pf_mean, pf_std = _perfold_stats(d)
        if pf_mean is None and results:
            pf_mean = (results.get("test_metrics") or {}).get("f1_macro")
            if pf_mean is None:
                pf_mean = results.get("mean_f1")
        if pf_std is None and results:
            pf_std = results.get("std_f1")

        experiments.append(
            {
                "id": d.name,
                "name": config.get("name", d.name),
                "model_variant": config.get("model_variant") or (results.get("model_variant") if results else None),
                "type": exp_type,
                "status": "completed" if (results or exp_type == "baseline") else "partial",
                "num_folds": num_folds,
                "folds_completed": folds_done,
                "test_accuracy": test_accuracy,
                "mean_f1": pf_mean if (results or pf_mean is not None) else None,
                "test_f1": ((results.get("test_metrics") or {}).get("f1_macro") if results else None),
                "std_f1": pf_std if (results or pf_std is not None) else None,
                "timestamp": config.get("timestamp"),
                "is_ncd": is_ncd,
                **ncd_metrics,
            }
        )
    return experiments


def _perfold_stats(exp_dir) -> tuple[float | None, float | None]:
    """(mean, std) F1 test da perfold_test.json, o (None, None) se manca."""
    from app.core.perfold import load_perfold

    pf = load_perfold(exp_dir)
    if not pf:
        return None, None
    try:
        return float(pf["f1_mean"]), float(pf["f1_std"])
    except (ValueError, KeyError):
        return None, None


def _override_test_metrics(exp_dir, results: dict | None) -> None:
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


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(experiment_id: str):
    """Get full experiment detail with all folds. Prova DB, fallback a JSON."""
    experiment_id = _validate_id(experiment_id)
    exp_dir = _get_experiments_dir() / experiment_id

    # Fallback immediato se l'esperimento non esiste (prima di aprire il DB)
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    # Always use JSON files as source of truth (DB is read-only cache)
    # so we get complete fold data including supcon_phase

    config_path = exp_dir / "config.json"
    results_path = exp_dir / "results.json"

    if not config_path.exists() and not results_path.exists():
        raise HTTPException(status_code=404, detail=f"No config.json or results.json in '{experiment_id}'")

    config = _load_json(config_path) if config_path.exists() else {}

    # Merge recipe info from experiment definitions
    import re

    exp_num_match = re.search(r"exp(\d+)", experiment_id)
    if exp_num_match:
        exp_num = int(exp_num_match.group(1))
        _recipes = {
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
                "description": "Come Exp 3 ma O. garganica esclusa dal training "
                "(5 note): novel nascosta per OSR e GCD.",
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
                "  • Proiezione: MLP (embedding_dim → 128) scartata dopo Fase 1.\n"
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
                "  • SupCon standard + loss repulsiva SpHOR.\n"
                "  • Loss SpHOR: L_rep = (1/K²) Σ_{i≠j} cos(θ_ij)² — forza i prototipi "
                "ad essere ortogonali, massimizzando la copertura angolare.\n"
                "  • Previene il collasso dei prototipi e migliora la separazione open-set.\n"
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
                "  • Proiettato a 256-dim via MLP prima della SupCon loss.\n"
                "  • Stessi ottimizzatori e scheduling di Exp 3.\n\n"
                "Fase 2 — CE Fine-Tuning:\n"
                "  • Backbone congelato, testa lineare (come Exp 2-9).\n\n"
                "Valutazione: O. sphegodes_Napoli novel class (solo in test). "
                "Obiettivo: miglior rilevamento novel via embedding più separati.",
            },
        }
        if exp_num in _recipes:
            config["phases"] = _recipes[exp_num]["phases"]
            config["recipe"] = _recipes[exp_num]["recipe"]
            if _recipes[exp_num].get("description"):
                config["description"] = _recipes[exp_num]["description"]

    folds = []
    for fold_file in sorted(exp_dir.glob("fold_*.json")):
        raw = _load_json(fold_file)
        from app.api.routes.comparison import _normalize_fold

        folds.append(_normalize_fold(raw))

    # Fallback: se non ci sono fold file individuali, usa fold_results da results.json
    if not folds:
        results_path = exp_dir / "results.json"
        if results_path.exists():
            raw_results = _load_json(results_path)
            fr = raw_results.get("fold_results", [])
            if fr:
                from app.api.routes.comparison import _normalize_fold

                for f in fr:
                    folds.append(_normalize_fold(f))

    results = None
    results_path = exp_dir / "results.json"
    if results_path.exists():
        results = _load_json(results_path)

    # Test ufficiale: confusion/acc/F1 del best fold da perfold_test.json
    # (i test_metrics in results.json si riferiscono al vecchio test pre-split).
    _override_test_metrics(exp_dir, results)

    # NCD metrics from ncd_results.json (JSON fallback)
    ncd_full: dict | None = None
    ncd_path = exp_dir / "ncd_results.json"
    if ncd_path.exists():
        try:
            ncd_full = json.loads(ncd_path.read_text())
        except (json.JSONDecodeError, OSError):
            ncd_full = None
        ncd_metrics = _read_ncd_metrics(exp_dir)
        if ncd_metrics:
            results = results or {}
            results.update(ncd_metrics)

    last_modified = None
    rp = exp_dir / "results.json"
    if rp.exists():
        mtime = rp.stat().st_mtime
        last_modified = datetime.datetime.fromtimestamp(mtime).isoformat()

    return _build_experiment_response(
        exp_id=experiment_id,
        config=config,
        folds=folds,
        results=results,
        last_modified=last_modified,
        ncd_full=ncd_full,
    )


def _build_experiment_response(
    exp_id: str,
    config: dict,
    folds: list,
    results: dict | None = None,
    exp_data: dict | None = None,
    last_modified: str | None = None,
    ncd_full: dict | None = None,
) -> dict:
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
    pf_mean, pf_std = _perfold_stats(_get_experiments_dir() / exp_id)
    if pf_mean is None:
        pf_mean = (
            ((results or {}).get("test_metrics") or {}).get("f1_macro")
            if ((results or {}).get("test_metrics") or {}).get("f1_macro") is not None
            else (results or {}).get("mean_f1")
        )
    if pf_std is None:
        pf_std = (results or {}).get("std_f1")

    # Normalize all folds (handles SupCon two-phase → flat + phase_epochs)
    from app.api.routes.comparison import _normalize_fold

    normalized_folds = []
    for f in folds:
        if isinstance(f, dict):
            normalized_folds.append(_normalize_fold(f))
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


@router.get("/{experiment_id}/log")
async def get_experiment_log(experiment_id: str):
    """Get the training log file for an experiment.
    On HF Space (no log files synced), returns available=false.
    """
    experiment_id = _validate_id(experiment_id)
    exp_dir = _get_experiments_dir() / experiment_id
    log_file = exp_dir / f"{experiment_id}.log"
    if not log_file.exists():
        return {"log": None, "size": 0, "available": False}
    with open(log_file) as f:
        content = f.read()
    return {"log": content, "size": len(content), "available": True}


@router.get("/{experiment_id}/folds/{fold_number}", response_model=FoldDataResponse)
async def get_fold(experiment_id: str, fold_number: int):
    """Get a single fold with full training history.

    fold_number is 0-based (matching file names: fold_0.json, fold_1.json, ...).
    """
    experiment_id = _validate_id(experiment_id)
    exp_dir = _get_experiments_dir() / experiment_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    if fold_number < 0:
        raise HTTPException(status_code=400, detail="fold_number must be >= 0")

    fold_file = exp_dir / f"fold_{fold_number}.json"
    if not fold_file.exists():
        raise HTTPException(status_code=404, detail=f"Fold {fold_number} not found")

    from app.api.routes.comparison import _normalize_fold

    return _normalize_fold(_load_json(fold_file))
