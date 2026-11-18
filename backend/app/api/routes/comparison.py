import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ExperimentComparisonResponse,
    FoldSnapshot,
    ModelComparisonEntry,
    ModelComparisonResponse,
    SeriesModelEntry,
)
from app.api.serializers.folds import normalize_fold
from app.config import settings
from app.core.perfold import load_perfold


def _test_stats(exp_dir: Path) -> tuple[float | None, float | None]:
    pf = load_perfold(exp_dir)
    if not pf:
        return None, None
    try:
        return float(pf["f1_mean"]), float(pf["f1_std"])
    except (ValueError, KeyError):
        return None, None


router = APIRouter(prefix="/api/comparison", tags=["comparison"])

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 30


def _cache_get(key: str) -> object | None:
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < _CACHE_TTL:
        return _CACHE[key][1]
    return None


def _cache_set(key: str, value: object) -> None:
    _CACHE[key] = (time.time(), value)


_KNOWN_MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]

_EXPERIMENT_META = {
    1: {"label": "Exp 1", "description": "Baseline: train and test on private dataset", "leakage": True},
    2: {"label": "Exp 2", "description": "Domain shift: train private, test internet raw", "leakage": True},
    3: {"label": "Exp 3", "description": "Domain shift: train private, test internet cropped", "leakage": True},
    4: {"label": "Exp 4", "description": "Combined dataset, test internet cropped", "leakage": True},
    5: {"label": "Exp 5", "description": "Combined dataset, test private (catastrophic forgetting)", "leakage": True},
    6: {"label": "Exp 1", "description": "Clean split, no leakage, 6 architectures", "leakage": False},
    7: {"label": "Exp 2 (SupCon)", "description": "Supervised Contrastive + CE fine-tuning", "leakage": False},
}


def _detect_experiment_id(dir_name: str) -> int | None:
    m = re.match(r"exp(\d+)", dir_name)
    return int(m.group(1)) if m else None


def _detect_model_variant(dir_name: str) -> str | None:
    for model in _KNOWN_MODELS:
        if dir_name.endswith(model):
            return model
    return None


_DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "dinov2_small": "DINOv2-Small",
    "dinov2_base": "DINOv2-Base",
}


def _experiment_dir(model_variant: str) -> Path:
    prefix = settings.experiment_comparison_prefix
    return Path(settings.results_dir) / f"{prefix}_{model_variant}"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            data: Any = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


@router.get("", response_model=ModelComparisonResponse)
async def get_comparison() -> ModelComparisonResponse:
    cached = _cache_get("comparison")
    if isinstance(cached, ModelComparisonResponse):
        return cached

    exp_dir = Path(settings.results_dir)
    if not exp_dir.exists():
        return ModelComparisonResponse(models=[])

    prefix = settings.experiment_comparison_prefix
    models: list[ModelComparisonEntry] = []

    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue

        results = _load_json(d / "results.json")
        if not results:
            continue

        config = _load_json(d / "config.json") or {}
        variant = config.get("model_variant", d.name.replace(prefix, ""))
        display_name = _DISPLAY_NAMES.get(variant, variant)

        fold_snapshots: list[FoldSnapshot] = []
        for ff in sorted(d.glob("fold_*.json")):
            fdata = _load_json(ff)
            if fdata:
                fold_snapshots.append(
                    FoldSnapshot(
                        fold=fdata.get("fold", 0),
                        best_f1=fdata.get("best_f1"),
                        best_accuracy=fdata.get("best_accuracy"),
                        best_epoch=fdata.get("best_epoch"),
                    )
                )

        _tmean, _tstd = _test_stats(d)
        models.append(
            ModelComparisonEntry(
                name=display_name,
                model_variant=variant,
                folds_completed=len(fold_snapshots),
                num_folds=config.get("num_folds", 5),
                mean_f1=_tmean
                if _tmean is not None
                else (
                    results.get("mean_f1")
                    if results.get("mean_f1") is not None
                    else results.get("test_metrics", {}).get("f1_macro")
                ),
                std_f1=_tstd if _tstd is not None else results.get("std_f1"),
                mean_accuracy=results.get("mean_accuracy"),
                test_accuracy=results.get("test_metrics", {}).get("accuracy"),
                test_f1=results.get("test_metrics", {}).get("f1_macro"),
                best_fold=results.get("best_fold"),
                fold_results=fold_snapshots,
            )
        )

    models.sort(key=lambda m: (-(m.mean_f1 if m.mean_f1 is not None else 0), m.model_variant))
    for i, m in enumerate(models):
        m.rank = i + 1

    result = ModelComparisonResponse(models=models)
    _cache_set("comparison", result)
    return result


def _build_exp_group(prefix: str) -> list[ModelComparisonEntry]:
    exp_dir = Path(settings.results_dir)
    if not exp_dir.exists():
        return []

    models: list[ModelComparisonEntry] = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        results = _load_json(d / "results.json")
        if not results:
            continue
        config = _load_json(d / "config.json") or {}
        variant = config.get("model_variant", d.name.replace(prefix + "_", ""))
        display_name = _DISPLAY_NAMES.get(variant, variant)

        _tmean2, _tstd2 = _test_stats(d)
        models.append(
            ModelComparisonEntry(
                name=display_name,
                model_variant=variant,
                folds_completed=0,
                num_folds=config.get("num_folds", 5),
                mean_f1=_tmean2
                if _tmean2 is not None
                else (
                    results.get("mean_f1")
                    if results.get("mean_f1") is not None
                    else results.get("test_metrics", {}).get("f1_macro")
                ),
                std_f1=_tstd2 if _tstd2 is not None else results.get("std_f1"),
                mean_accuracy=results.get("mean_accuracy"),
                test_accuracy=results.get("test_metrics", {}).get("accuracy"),
                test_f1=results.get("test_metrics", {}).get("f1_macro"),
                best_fold=results.get("best_fold"),
            )
        )

    models.sort(key=lambda m: (-(m.mean_f1 if m.mean_f1 is not None else 0), m.model_variant))
    for i, m in enumerate(models):
        m.rank = i + 1
    return models


@router.get("/experiments", response_model=ExperimentComparisonResponse)
async def compare_experiments() -> ExperimentComparisonResponse:
    cached = _cache_get("experiments")
    if isinstance(cached, ExperimentComparisonResponse):
        return cached
    exp1 = _build_exp_group("exp1_ce")
    exp2 = _build_exp_group("exp2_supcon")
    exp3 = _build_exp_group("exp3_sphor")

    # Build delta for exp1→exp2→exp3
    all_models = list({m.model_variant for m in exp1 + exp2 + exp3})
    delta: list[SeriesModelEntry] = []
    for variant in sorted(all_models):
        m1 = next((m for m in exp1 if m.model_variant == variant), None)
        m2 = next((m for m in exp2 if m.model_variant == variant), None)
        m3 = next((m for m in exp3 if m.model_variant == variant), None)
        delta.append(
            SeriesModelEntry(
                variant=variant,
                display_name=_DISPLAY_NAMES.get(variant, variant),
                f1_ce=m1.mean_f1 if m1 else None,
                f1_supcon=m2.mean_f1 if m2 else None,
                f1_sphor=m3.mean_f1 if m3 else None,
            )
        )

    result = ExperimentComparisonResponse(
        exp1=exp1,
        exp2=exp2,
        exp3=exp3,
        delta=delta,
    )
    _cache_set("experiments", result)
    return result


@router.get("/{model_variant}/fold/{fold_number}")
async def get_comparison_fold(model_variant: str, fold_number: int) -> dict[str, Any]:
    if model_variant not in _KNOWN_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model variant: {model_variant}")
    exp_dir = _experiment_dir(model_variant)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment for '{model_variant}' not found")

    fold_file = exp_dir / f"fold_{fold_number}.json"
    if not fold_file.exists():
        raise HTTPException(status_code=404, detail=f"Fold {fold_number} not found for {model_variant}")

    raw = _load_json(fold_file)
    if raw is None:
        raise HTTPException(status_code=500, detail=f"Fold {fold_number} illeggibile per {model_variant}")
    return normalize_fold(raw)
