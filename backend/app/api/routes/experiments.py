# backend/app/api/routes/experiments.py
import datetime
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import ExperimentDetailResponse, ExperimentSummary, FoldDataResponse
from app.config import settings

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
    'exp10_mgh_ncd_resnet18' → 10, 'exp6_clean_split' → 6."""
    m = re.search(r'exp(\d+)', name)
    return int(m.group(1)) if m else 999


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


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

        # Nascondi exp9 (non rilevante)
        if "exp9_ncd" in d.name:
            continue

        config_path = d / "config.json"
        results_path = d / "results.json"

        # Need at least config.json or results.json
        if not config_path.exists() and not results_path.exists():
            continue

        config = _load_json(config_path) if config_path.exists() else {}
        results = _load_json(results_path) if results_path.exists() else None

        fold_files = sorted(d.glob("fold_*.json"))
        has_folds = len(fold_files) > 0
        config_type = (config or {}).get("type", "")
        exp_type = "kfold" if has_folds else ("kfold" if config_type == "kfold" else "baseline")

        experiments.append(
            {
                "id": d.name,
                "name": config.get("name", d.name),
                "model_variant": config.get("model_variant"),
                "type": exp_type,
                "status": "completed" if (results or exp_type == "baseline") else "partial",
                "num_folds": len(fold_files) if has_folds else (config or {}).get("num_folds", 5),
                "folds_completed": len(fold_files),
                "test_accuracy": results.get("test_metrics", {}).get("accuracy") if results else None,
                "mean_f1": (results.get("mean_f1") or results.get("test_metrics", {}).get("f1_macro"))
                if results
                else None,
                "std_f1": results.get("std_f1") if results else None,
                "timestamp": config.get("timestamp"),
            }
        )

    return experiments


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(experiment_id: str):
    """Get full experiment detail with all folds. Prova DB, fallback a JSON."""
    experiment_id = _validate_id(experiment_id)
    exp_dir = _get_experiments_dir() / experiment_id

    # Try DB first
    try:
        from app.core.db import get_db, _row_to_dict

        conn = get_db()
        try:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row:
                exp_data = _row_to_dict(row)
                folds_rows = conn.execute(
                    "SELECT * FROM folds WHERE experiment_id = ? ORDER BY fold_idx",
                    (experiment_id,),
                ).fetchall()
                folds = [_row_to_dict(r) for r in folds_rows]

                config = {
                    k: exp_data.get(k) for k in [
                        "model_variant", "num_folds", "num_classes", "loss_type",
                        "phases", "recipe",
                    ]
                }
                config = {k: v for k, v in config.items() if v is not None}

                return _build_experiment_response(
                    exp_id=experiment_id,
                    config=config,
                    folds=folds,
                    exp_data=exp_data,
                )
        finally:
            conn.close()
    except Exception as e:
        logger.warning("DB fallito per %s — fallback a JSON: %s", experiment_id, e)

    # Fallback a JSON filesystem
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    config_path = exp_dir / "config.json"
    results_path = exp_dir / "results.json"

    if not config_path.exists() and not results_path.exists():
        raise HTTPException(status_code=404, detail=f"No config.json or results.json in '{experiment_id}'")

    config = _load_json(config_path) if config_path.exists() else {}

    # Merge recipe info from experiment definitions
    import re
    exp_num_match = re.search(r'exp(\d+)', experiment_id)
    if exp_num_match:
        exp_num = int(exp_num_match.group(1))
        _recipes = {
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
                          "  • Loss: SupConLoss — attrae embedding della stessa specie, respinge specie diverse sull'ipersfera.\n"
                          "  • Temperatura τ = 0.07 (valore standard da [8]).\n"
                          "  • Proiezione: MLP (embedding_dim → 128) scartata dopo Fase 1.\n"
                          "  • Backbone + projection head addestrati insieme.\n"
                          "  • Metrica di validazione: KNN accuracy (k=1).\n"
                          "  • Ottimizzatore: AdamW per-architettura (ResNet: lr=0.001, ConvNeXt: lr=0.0001, DINOv2: lr=0.0002).\n\n"
                          "Fase 2 — CE Fine-Tuning (Linear Probe):\n"
                          "  • Backbone congelato (preserva struttura contrastiva).\n"
                          "  • Solo testa lineare addestrata con Cross-Entropy.\n"
                          "  • LR più alto (0.01) perché solo la testa casuale si addestra.\n"
                          "  • Adam, max 50 epoche, early stopping (patience=15).",
            },
            8: {
                "phases": ["Fase 1: SupCon + SpHOR", "Fase 2: CE fine-tuning"],
                "recipe": "Stessa struttura a due fasi di Exp 7, con regolarizzazione aggiuntiva in Fase 1.\n\n"
                          "Fase 1 — SupCon + SpHOR (Prototipi Ortogonali Sferici):\n"
                          "  • SupCon standard + loss repulsiva SpHOR.\n"
                          "  • Loss SpHOR: L_rep = (1/K²) Σ_{i≠j} cos(θ_ij)² — forza i prototipi "
                          "ad essere ortogonali, massimizzando la copertura angolare.\n"
                          "  • Previene il collasso dei prototipi e migliora la separazione open-set.\n"
                          "  • Stessi ottimizzatori e scheduling di Exp 7.\n\n"
                          "Fase 2 — CE Fine-Tuning:\n"
                          "  • Identica a Exp 7: backbone congelato, testa lineare.\n\n"
                          "Valutazione: O. sphegodes_Napoli è la novel class (solo in test).",
            },
            9: {
                "phases": ["Fase 1: SupCon + SpHOR", "Fase 2: CE fine-tuning"],
                "recipe": "Strutturalmente identico a Exp 8 (stessa loss, stessi iperparametri).\n"
                          "L'unica differenza: ncd_mode=True abilita il pipeline NCD completo "
                          "(OSR + clustering) con O. sphegodes_Napoli come novel class nascosta.",
            },
            10: {
                "phases": ["Fase 1: MGH-SupCon (Multi-Granularità Hard-negative)", "Fase 2: CE fine-tuning"],
                "recipe": "Training a due fasi potenziato con due innovazioni chiave rispetto a Exp 8.\n\n"
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
                          "  • Stessi ottimizzatori e scheduling di Exp 8.\n\n"
                          "Fase 2 — CE Fine-Tuning:\n"
                          "  • Backbone congelato, testa lineare (come Exp 7-9).\n\n"
                          "Valutazione: O. sphegodes_Napoli novel class (solo in test). "
                          "Obiettivo: miglior rilevamento novel via embedding più separati.",
            },
        }
        if exp_num in _recipes:
            config["phases"] = _recipes[exp_num]["phases"]
            config["recipe"] = _recipes[exp_num]["recipe"]

    folds = []
    for fold_file in sorted(exp_dir.glob("fold_*.json")):
        raw = _load_json(fold_file)
        from app.api.routes.comparison import _normalize_fold
        folds.append(_normalize_fold(raw))

    results = None
    results_path = exp_dir / "results.json"
    if results_path.exists():
        results = _load_json(results_path)

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
    )


def _build_experiment_response(
    exp_id: str,
    config: dict,
    folds: list,
    results: dict | None = None,
    exp_data: dict | None = None,
    last_modified: str | None = None,
) -> dict:
    """Costruisce risposta standard per un esperimento."""
    if exp_data:
        results = {
            "mean_f1": exp_data.get("mean_f1"),
            "std_f1": exp_data.get("std_f1"),
            "mean_accuracy": exp_data.get("mean_accuracy"),
            "test_metrics": {
                "f1_macro": exp_data.get("test_f1"),
                "accuracy": exp_data.get("test_accuracy"),
            } if exp_data.get("test_f1") else None,
            "best_fold": exp_data.get("best_fold"),
        }
    return {
        "id": exp_id,
        "config": config,
        "folds": folds,
        "results": results,
        "last_modified": last_modified,
        "summary": {
            "model_variant": config.get("model_variant"),
            "type": "kfold" if folds else "baseline",
            "num_folds": len(folds),
            "mean_accuracy": (results or {}).get("mean_accuracy"),
            "mean_f1": ((results or {}).get("mean_f1") or
                       ((results or {}).get("test_metrics") or {}).get("f1_macro")),
            "std_f1": (results or {}).get("std_f1"),
            "test_accuracy": ((results or {}).get("test_metrics") or {}).get("accuracy"),
            "test_f1": ((results or {}).get("test_metrics") or {}).get("f1_macro"),
        },
    }


@router.get("/{experiment_id}/log")
async def get_experiment_log(experiment_id: str):
    """Get the training log file for an experiment.
    On HF Space (no log files synced), returns available=false."""
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

    return _load_json(fold_file)
