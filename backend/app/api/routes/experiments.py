# backend/app/api/routes/experiments.py
import json
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
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir():
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
                "num_folds": len(fold_files) if has_folds else None,
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
    """Get full experiment detail with all folds."""
    experiment_id = _validate_id(experiment_id)
    exp_dir = _get_experiments_dir() / experiment_id
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    config_path = exp_dir / "config.json"
    results_path = exp_dir / "results.json"

    if not config_path.exists() and not results_path.exists():
        raise HTTPException(status_code=404, detail=f"No config.json or results.json in '{experiment_id}'")

    config = _load_json(config_path) if config_path.exists() else {}

    folds = []
    for fold_file in sorted(exp_dir.glob("fold_*.json")):
        folds.append(_load_json(fold_file))

    results = None
    results_path = exp_dir / "results.json"
    if results_path.exists():
        results = _load_json(results_path)

    import datetime
    last_modified = None
    rp = exp_dir / "results.json"
    if rp.exists():
        mtime = rp.stat().st_mtime
        last_modified = datetime.datetime.fromtimestamp(mtime).isoformat()

    return {
        "id": experiment_id,
        "config": config,
        "folds": folds,
        "results": results,
        "last_modified": last_modified,
        "summary": {
            "model_variant": config.get("model_variant"),
            "type": "kfold" if folds else "baseline",
            "num_folds": len(folds),
            "mean_accuracy": results.get("mean_accuracy") if results else None,
            "mean_f1": (results.get("mean_f1") or results.get("test_metrics", {}).get("f1_macro")) if results else None,
            "std_f1": results.get("std_f1") if results else None,
            "test_accuracy": results.get("test_metrics", {}).get("accuracy") if results else None,
            "test_f1": results.get("test_metrics", {}).get("f1_macro") if results else None,
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
