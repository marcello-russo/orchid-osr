import json
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ExperimentComparisonResponse,
    FoldSnapshot,
    ModelComparisonEntry,
    ModelComparisonResponse,
    SeriesModelEntry,
)
from app.config import settings

router = APIRouter(prefix="/api/comparison", tags=["comparison"])

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 30


def _cache_get(key: str):
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < _CACHE_TTL:
        return _CACHE[key][1]
    return None


def _cache_set(key: str, value: object):
    _CACHE[key] = (time.time(), value)


_KNOWN_MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]

_EXPERIMENT_META = {
    1: {"label": "Exp 1", "description": "Baseline: train and test on private dataset", "leakage": True},
    2: {"label": "Exp 2", "description": "Domain shift: train private, test internet raw", "leakage": True},
    3: {"label": "Exp 3", "description": "Domain shift: train private, test internet cropped", "leakage": True},
    4: {"label": "Exp 4", "description": "Combined dataset, test internet cropped", "leakage": True},
    5: {"label": "Exp 5", "description": "Combined dataset, test private (catastrophic forgetting)", "leakage": True},
    6: {"label": "Exp 6", "description": "Clean split, no leakage, 6 architectures", "leakage": False},
    7: {"label": "Exp 7 (SupCon)", "description": "Supervised Contrastive + CE fine-tuning", "leakage": False},
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


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@router.get("", response_model=ModelComparisonResponse)
async def get_comparison():
    cached = _cache_get("comparison")
    if cached:
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

        models.append(
            ModelComparisonEntry(
                name=display_name,
                model_variant=variant,
                folds_completed=len(fold_snapshots),
                num_folds=config.get("num_folds", 5),
                mean_f1=results.get("mean_f1")
                if results.get("mean_f1") is not None
                else results.get("test_metrics", {}).get("f1_macro"),
                std_f1=results.get("std_f1"),
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

        models.append(
            ModelComparisonEntry(
                name=display_name,
                model_variant=variant,
                folds_completed=0,
                num_folds=config.get("num_folds", 5),
                mean_f1=results.get("mean_f1")
                if results.get("mean_f1") is not None
                else results.get("test_metrics", {}).get("f1_macro"),
                std_f1=results.get("std_f1"),
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
async def compare_experiments():
    cached = _cache_get("experiments")
    if cached:
        return cached
    exp6 = _build_exp_group("exp6_clean_split")
    exp7 = _build_exp_group("exp7_supcon")
    exp8 = _build_exp_group("exp8_sphor")

    # Build delta for exp6→exp7→exp8
    all_models = list({m.model_variant for m in exp6 + exp7 + exp8})
    delta = []
    for variant in sorted(all_models):
        m6 = next((m for m in exp6 if m.model_variant == variant), None)
        m7 = next((m for m in exp7 if m.model_variant == variant), None)
        m8 = next((m for m in exp8 if m.model_variant == variant), None)
        delta.append(
            SeriesModelEntry(
                variant=variant,
                display_name=_DISPLAY_NAMES.get(variant, variant),
                f1_ce=m6.mean_f1 if m6 else None,
                f1_supcon=m7.mean_f1 if m7 else None,
                f1_sphor=m8.mean_f1 if m8 else None,
            )
        )

    result = ExperimentComparisonResponse(
        exp6=exp6,
        exp7=exp7,
        exp8=exp8,
        delta=delta,
    )
    _cache_set("experiments", result)
    return result


def _normalize_fold(fold: dict) -> dict:
    """Normalize fold data, separating SupCon and CE phases into fold_phases.

    If fold already has fold_phases (already normalized), return as-is.
    Single-phase (CE baseline, exp6):
      - Truncates history to best_epoch + patience
      - fold_phases = None

    Two-phase (SupCon, exp7+):
      - Phase 1 (contrastive): supcon_history metrics
      - Phase 2 (CE): top-level history metrics (truncated)
      - Returns fold_phases array with Phase 1 data
    """
    if "fold_phases" in fold:
        return fold
    if "supcon_phase" not in fold:
        # Single phase (CE baseline, exp6) — truncate history
        flat = dict(fold)
        total = len(flat.get("history", {}).get("train_loss", []))
        best = flat.get("best_epoch", 0)
        cap = min(best, total - 1) if best > 0 and total > 0 else 0
        cut = min(cap + 15, total) if cap > 0 else total
        if cap > 0 and cut < cap + 1:
            cut = min(cap + 1, total)
        if cut < total:
            h = flat.get("history", {})
            flat["history"] = {k: v[:cut] for k, v in h.items() if isinstance(v, list)}
            flat["best_epoch"] = cut - 1
        flat["fold_phases"] = None
        return flat

    # Two-phase (SupCon, exp7+)
    sp = fold["supcon_phase"]
    sh = sp.get("supcon_history", {})
    ce_hist = fold.get("history", {})

    # Phase 1: SupCon
    supcon_train = sh.get("train_loss", []) or []
    supcon_knn = sh.get("val_knn_acc", []) or []

    # Phase 2: CE
    ce_train = ce_hist.get("train_loss", []) or []
    ce_val_loss = ce_hist.get("val_loss", []) or []
    ce_val_f1 = ce_hist.get("val_f1", []) or []
    ce_val_acc = ce_hist.get("val_accuracy", []) or []

    # Truncate Phase 2 to best_epoch + patience
    n_ce = len(ce_train)
    best_raw = fold.get("best_epoch", 0)
    ce_cap = min(best_raw, n_ce - 1) if best_raw > 0 and n_ce > 0 else 0
    ce_cut = min(ce_cap + 15, n_ce) if ce_cap > 0 else n_ce
    if ce_cap > 0 and ce_cut < ce_cap + 1:
        ce_cut = min(ce_cap + 1, n_ce)

    ce_history = {
        "train_loss": ce_train[:ce_cut],
        "val_loss": ce_val_loss[:ce_cut],
        "val_f1": ce_val_f1[:ce_cut],
        "val_accuracy": ce_val_acc[:ce_cut],
    }
    best_epoch = ce_cut - 1 if ce_cut > 0 else 0

    fold_phases = [
        {
            "name": "SupCon",
            "type": "contrastive",
            "epochs": len(supcon_train),
            "metrics": {
                "train_loss": supcon_train,
                "val_knn_acc": supcon_knn,
            },
        }
    ]

    result = {
        "fold": fold.get("fold", 0),
        "best_f1": fold.get("best_f1", 0),
        "best_accuracy": fold.get("best_accuracy", 0),
        "best_epoch": best_epoch,
        "final_train_loss": ce_history["train_loss"][-1] if ce_history["train_loss"] else 0,
        "final_val_loss": ce_history["val_loss"][-1] if ce_history["val_loss"] else 0,
        "history": ce_history,
        "train_size": fold.get("train_size", 0),
        "val_size": fold.get("val_size", 0),
        "fold_phases": fold_phases,
    }
    # Preserve optional fields from original fold
    for key in ("phase_epochs", "phase_names"):
        if key in fold:
            result[key] = fold[key]
    return result


@router.get("/{model_variant}/fold/{fold_number}")
async def get_comparison_fold(model_variant: str, fold_number: int):
    if model_variant not in _KNOWN_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model variant: {model_variant}")
    exp_dir = _experiment_dir(model_variant)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment for '{model_variant}' not found")

    fold_file = exp_dir / f"fold_{fold_number}.json"
    if not fold_file.exists():
        raise HTTPException(status_code=404, detail=f"Fold {fold_number} not found for {model_variant}")

    raw = _load_json(fold_file)
    return _normalize_fold(raw)
