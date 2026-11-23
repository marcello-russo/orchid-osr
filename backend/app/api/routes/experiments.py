# backend/app/api/routes/experiments.py
"""Route degli esperimenti: I/O e validazione; la serializzazione vive nei serializers."""

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import ExperimentDetailResponse, ExperimentSummary, FoldDataResponse
from app.api.serializers.experiments import (
    RECIPES,
    build_experiment_response,
    load_json,
    override_test_metrics,
    perfold_stats,
    read_ncd_metrics,
)
from app.api.serializers.folds import normalize_fold
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


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments() -> list[dict[str, Any]]:
    """List all experiments with summary metrics."""
    exp_dir = _get_experiments_dir()
    if not exp_dir.exists():
        return []

    experiments: list[dict[str, Any]] = []
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

        config = load_json(config_path) if config_path.exists() else {}
        results = load_json(results_path) if results_path.exists() else None
        # stessi test_metrics del detail (best fold sul test ufficiale)
        override_test_metrics(d, results)

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
        ncd_metrics = read_ncd_metrics(d) if has_ncd_file else {}

        test_accuracy = results.get("test_metrics", {}).get("accuracy") if results else None

        # F1 test dal test ufficiale: medie ± std sui 5 fold (perfold_test.json).
        # Fallback al vecchio comportamento solo se il file manca.
        pf_mean, pf_std = perfold_stats(d)
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


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(experiment_id: str) -> dict[str, Any]:
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

    config = load_json(config_path) if config_path.exists() else {}

    # Merge recipe info from experiment definitions
    exp_num_match = re.search(r"exp(\d+)", experiment_id)
    if exp_num_match:
        recipe = RECIPES.get(int(exp_num_match.group(1)))
        if recipe:
            config["phases"] = recipe["phases"]
            config["recipe"] = recipe["recipe"]
            if recipe.get("description"):
                config["description"] = recipe["description"]

    folds = []
    for fold_file in sorted(exp_dir.glob("fold_*.json")):
        raw = load_json(fold_file)
        folds.append(normalize_fold(raw))

    # Fallback: se non ci sono fold file individuali, usa fold_results da results.json
    if not folds:
        results_path = exp_dir / "results.json"
        if results_path.exists():
            raw_results = load_json(results_path)
            fr = raw_results.get("fold_results", [])
            if fr:
                for f in fr:
                    folds.append(normalize_fold(f))

    results = None
    results_path = exp_dir / "results.json"
    if results_path.exists():
        results = load_json(results_path)

    # Test ufficiale: confusion/acc/F1 del best fold da perfold_test.json
    # (i test_metrics in results.json si riferiscono al vecchio test pre-split).
    override_test_metrics(exp_dir, results)

    # NCD metrics from ncd_results.json (JSON fallback)
    ncd_full: dict[str, Any] | None = None
    ncd_path = exp_dir / "ncd_results.json"
    if ncd_path.exists():
        try:
            ncd_full = json.loads(ncd_path.read_text())
        except (json.JSONDecodeError, OSError):
            ncd_full = None
        ncd_metrics = read_ncd_metrics(exp_dir)
        if ncd_metrics:
            results = results or {}
            results.update(ncd_metrics)

    last_modified = None
    rp = exp_dir / "results.json"
    if rp.exists():
        mtime = rp.stat().st_mtime
        last_modified = datetime.datetime.fromtimestamp(mtime).isoformat()

    return build_experiment_response(
        exp_id=experiment_id,
        exp_dir=exp_dir,
        config=config,
        folds=folds,
        results=results,
        last_modified=last_modified,
        ncd_full=ncd_full,
    )


@router.get("/{experiment_id}/log")
async def get_experiment_log(experiment_id: str) -> dict[str, Any]:
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
async def get_fold(experiment_id: str, fold_number: int) -> dict[str, Any]:
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

    return normalize_fold(load_json(fold_file))
