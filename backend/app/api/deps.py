# app/api/deps.py
import json
import logging
from collections import OrderedDict
from pathlib import Path

import torch

from app.api.schemas import ModelOption
from app.config import settings
from app.core.models import ClassifierModel, available_models, create_model

logger = logging.getLogger(__name__)

MAX_CACHED_MODELS = 5

# Class names from centralized config
CLASS_NAMES = settings.class_names


def _get_device() -> torch.device:
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Public alias for backward compatibility
get_device = _get_device


def _safe_json_load(path: Path) -> dict | None:
    """Load JSON with error handling."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _get_mean_f1(results: dict) -> float | None:
    """Get mean_f1 with fallback to test_metrics.f1_macro."""
    return results.get("mean_f1") or results.get("test_metrics", {}).get("f1_macro")


def _format_model_name(exp_id: str, model_variant: str) -> str:
    """Generate a user-friendly model name from experiment directory name."""
    variant_display = {
        "resnet18": "ResNet-18",
        "resnet50": "ResNet-50",
        "convnext_tiny": "ConvNeXt-Tiny",
        "convnext_small": "ConvNeXt-Small",
        "dinov2_small": "DINOv2-Small",
        "dinov2_base": "DINOv2-Base",
    }.get(model_variant, model_variant)

    if "supcon" in exp_id or "sphor" in exp_id:
        exp_label = "Exp 7 (SupCon)" if "supcon" in exp_id else "Exp 8 (SpHOR)"
    else:
        exp_label = "Exp 6 (CE)"

    return f"{variant_display} — {exp_label}"


def _discover_models() -> list[ModelOption]:
    """Discover all trained models with weights and metrics."""
    experiments_dir = Path(settings.results_dir)
    options: list[ModelOption] = []

    for results_json in sorted(experiments_dir.rglob("results.json")):
        results = _safe_json_load(results_json)
        if results is None:
            continue

        parent = results_json.parent
        best_fold = results.get("best_fold")
        if best_fold is None:
            continue
        candidate = parent / f"best_fold_{best_fold}.pt"
        if not candidate.exists():
            continue

        exp_id = parent.name
        mv = results.get("model_variant", "resnet18")
        tm = results.get("test_metrics", {})

        options.append(ModelOption(
            id=exp_id,
            name=_format_model_name(exp_id, mv),
            model_variant=mv,
            mean_f1=_get_mean_f1(results),
            test_accuracy=tm.get("accuracy") if tm else None,
            num_folds=results.get("num_folds"),
        ))

    options.sort(key=lambda m: m.mean_f1 or 0, reverse=True)
    return options


def get_model_options() -> list[ModelOption]:
    """Get list of available models for classification."""
    return _discover_models()


def _validate_model_id(model_id: str) -> str:
    """Sanitize model_id to prevent path traversal."""
    if ".." in model_id or "/" in model_id or "\\" in model_id:
        raise ValueError(f"Invalid model_id: {model_id}")
    return model_id


def _find_weights(model_type: str) -> Path | None:
    """Find best available trained weights for a model type."""
    experiments_dir = Path(settings.results_dir)
    models_dir = Path(settings.models_dir)

    best_score = -1.0
    best_path: Path | None = None
    for results_json in experiments_dir.rglob(f"*_kfold_{model_type}/results.json"):
        results = _safe_json_load(results_json)
        if results is None:
            continue

        parent = results_json.parent
        best_fold = results.get("best_fold")
        if best_fold is not None:
            candidate = parent / f"best_fold_{best_fold}.pt"
            if candidate.exists():
                score = _get_mean_f1(results) or 0
                if score > best_score:
                    best_score = score
                    best_path = candidate

    if best_path:
        logger.info(f"Loading best k-fold weights: {best_path} (F1={best_score:.4f})")
        return best_path

    for subdir in sorted(models_dir.glob(f"train_*_{model_type}")):
        model_path = subdir / "model.pt"
        if model_path.exists():
            logger.info(f"Loading trained weights: {model_path}")
            return model_path

    return None


def _find_weights_by_id(model_id: str) -> tuple[Path, str] | None:
    """Find weights for a specific experiment ID. Returns (path, model_variant)."""
    model_id = _validate_model_id(model_id)
    experiments_dir = Path(settings.results_dir)
    results_json = experiments_dir / model_id / "results.json"
    if not results_json.exists():
        return None

    results = _safe_json_load(results_json)
    if results is None:
        return None

    best_fold = results.get("best_fold")
    if best_fold is None:
        return None

    candidate = results_json.parent / f"best_fold_{best_fold}.pt"
    if not candidate.exists():
        return None

    return candidate, results.get("model_variant", "resnet18")


# LRU-style cache for loaded models
_model_cache: OrderedDict[str, ClassifierModel] = OrderedDict()


def get_model(model_type: str = "resnet18", model_id: str | None = None) -> ClassifierModel:
    """Load and cache model. If model_id given, load that specific experiment's weights."""
    if model_id:
        result = _find_weights_by_id(model_id)
        if result is None:
            raise ValueError(f"No weights found for experiment: {model_id}")
        weights_path, variant = result
        cache_key = f"{model_id}:{weights_path}"
    else:
        valid = available_models()
        if model_type not in valid:
            raise ValueError(f"Unknown model type: {model_type}. Valid: {valid}")
        variant = model_type
        weights_path = _find_weights(model_type)
        cache_key = f"{model_type}:{weights_path}"

    if cache_key in _model_cache:
        _model_cache.move_to_end(cache_key)
        return _model_cache[cache_key]

    model = create_model(variant, num_classes=settings.num_classes, pretrained=True)

    if weights_path:
        model.load_weights(str(weights_path))
        logger.info(f"Loaded trained weights from {weights_path}")
    else:
        logger.warning(f"No trained weights found for {model_type}, using ImageNet pretrained")

    model.eval()
    device = _get_device()
    model = model.to(device)

    if len(_model_cache) >= MAX_CACHED_MODELS:
        evicted_key, evicted = _model_cache.popitem(last=False)
        logger.info(f"Evicted model from cache: {evicted_key}")

    _model_cache[cache_key] = model
    return model
