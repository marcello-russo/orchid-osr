"""Model discovery, caching, and loading with HuggingFace Hub fallback.

JSON metadata (results.json, config.json) is always available in the
Docker image (tracked in git). Model weights (.pt) are downloaded from
HuggingFace Hub on demand and cached locally.
"""
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import torch
from fastapi import Depends, HTTPException, status, Request
from huggingface_hub import hf_hub_download

from app.core.auth import decode_jwt

from app.api.schemas import CacheStatus, ModelOption
from app.config import settings
from app.core.models import ClassifierModel, available_models, create_model

logger = logging.getLogger(__name__)

MAX_CACHED_MODELS = 5
CLASS_NAMES = settings.class_names
HF_MODELS_REPO = "marcellorusso/orchid-ncd-models"

# Mapping from experiment directory prefix → human label + number
_EXP_META: dict[str, tuple[str, int]] = {
    "exp6_clean_split": ("Exp 6 (CE)", 6),
    "exp7_supcon": ("Exp 7 (SupCon)", 7),
    "exp8_sphor": ("Exp 8 (SpHOR)", 8),
    "exp9_ncd": ("Exp 9 (NCD)", 9),
    "exp10_mgh_ncd": ("Exp 10 (MGH-NCD)", 10),
}

# Mapping from directory prefix → HF upload prefix (upload_models.py convention)
_HF_PREFIX_MAP: dict[str, str] = {
    "exp6_clean_split": "exp6_ce",
    "exp7_supcon": "exp7_supcon",
    "exp8_sphor": "exp8_sphor",
    "exp9_ncd": "exp9_ncd",
    "exp10_mgh_ncd": "exp10_mgh_ncd",
}

_VARIANT_DISPLAY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "dinov2_small": "DINOv2-Small",
    "dinov2_base": "DINOv2-Base",
}


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


get_device = _get_device


def _safe_json_load(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _get_mean_f1(results: dict) -> float | None:
    return results.get("mean_f1") or results.get("test_metrics", {}).get("f1_macro")


def _get_exp_prefix(dir_name: str) -> str | None:
    for prefix in _EXP_META:
        if dir_name.startswith(prefix):
            return prefix
    return None


def _format_model_name(dir_name: str, model_variant: str) -> str:
    variant = _VARIANT_DISPLAY.get(model_variant, model_variant)
    prefix = _get_exp_prefix(dir_name)
    if prefix:
        label, _ = _EXP_META[prefix]
        return f"{variant} — {label}"
    return variant


def _get_cache_dir() -> Path:
    cache = Path.home() / ".cache" / "orchid-weights"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _hf_path_for(dir_name: str, filename: str) -> str | None:
    prefix = _get_exp_prefix(dir_name)
    if prefix is None:
        return None
    hf_prefix = _HF_PREFIX_MAP.get(prefix, prefix)
    return f"{hf_prefix}/{dir_name}/{filename}"


def _estimate_pt_size(dir_name: str) -> float:
    """Estimate model file size in MB based on variant."""
    for variant, size in [
        ("dinov2_base", 320),
        ("dinov2_small", 160),
        ("convnext_small", 110),
        ("convnext_tiny", 110),
        ("resnet50", 100),
        ("resnet18", 45),
    ]:
        if variant in dir_name:
            return size
    return 100


# ── Model discovery ──────────────────────────────────────────

def _discover_models() -> list[ModelOption]:
    """Discover all trained models from local JSON metadata.

    JSON files (results.json, config.json) are always available because
    they are tracked in git and copied into the Docker image.
    Model weights (.pt) are downloaded on demand from HuggingFace Hub.
    """
    experiments_dir = Path(settings.results_dir)
    if not experiments_dir.exists():
        logger.warning(f"Experiments directory not found: {experiments_dir}")
        return []

    options: list[ModelOption] = []
    cache_dir = _get_cache_dir()

    for results_json in sorted(experiments_dir.rglob("results.json")):
        results = _safe_json_load(results_json)
        if results is None:
            continue

        parent = results_json.parent
        dir_name = parent.name
        best_fold = results.get("best_fold")
        if best_fold is None:
            continue

        prefix = _get_exp_prefix(dir_name)
        if prefix is None:
            continue

        exp_label, exp_num = _EXP_META[prefix]
        mv = results.get("model_variant", "resnet18")
        test_metrics = results.get("test_metrics", {})

        # Check if .pt is in local cache
        pt_name = f"best_fold_{best_fold}.pt"
        local_pt = cache_dir / f"{dir_name}_{pt_name}"
        is_cached = local_pt.exists()

        # Also check the original experiment dir (for local training)
        if not is_cached:
            orig_pt = parent / pt_name
            is_cached = orig_pt.exists()

        options.append(ModelOption(
            id=dir_name,
            name=_format_model_name(dir_name, mv),
            model_variant=mv,
            mean_f1=_get_mean_f1(results),
            test_accuracy=test_metrics.get("accuracy"),
            num_folds=results.get("num_folds"),
            is_cached=is_cached,
            size_mb=_estimate_pt_size(dir_name),
            exp_num=exp_num,
            exp_label=exp_label,
        ))

    options.sort(key=lambda m: (-m.exp_num, -(m.mean_f1 or 0)))
    return options


def get_model_options() -> list[ModelOption]:
    return _discover_models()


# ── Weight loading with HF fallback ──────────────────────────

def _validate_model_id(model_id: str) -> str:
    if ".." in model_id or "/" in model_id or "\\" in model_id:
        raise ValueError(f"Invalid model_id: {model_id}")
    return model_id


def _resolve_weights(model_id: str) -> tuple[Path, str] | None:
    """Find or download weights for a specific experiment.

    Priority:
    1. Local experiment dir (training machine)
    2. Local cache dir (previously downloaded)
    3. Download from HuggingFace Hub
    """
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

    variant = results.get("model_variant", "resnet18")
    pt_name = f"best_fold_{best_fold}.pt"

    # 1. Local experiment dir
    local_pt = results_json.parent / pt_name
    if local_pt.exists():
        return local_pt, variant

    # 2. Local cache dir
    cache_pt = _get_cache_dir() / f"{model_id}_{pt_name}"
    if cache_pt.exists():
        return cache_pt, variant

    # 3. Download from HuggingFace Hub
    hf_path = _hf_path_for(model_id, pt_name)
    if hf_path is None:
        logger.error(f"Cannot resolve HF path for {model_id}")
        return None

    try:
        downloaded = hf_hub_download(
            repo_id=HF_MODELS_REPO,
            filename=hf_path,
            token=settings.hf_token or None,
        )
        # Copy to cache
        cache_pt = _get_cache_dir() / f"{model_id}_{pt_name}"
        if not cache_pt.exists():
            import shutil
            shutil.copy2(downloaded, cache_pt)
        logger.info(f"Downloaded weights from HF: {hf_path} ({downloaded})")
        return cache_pt, variant
    except Exception as e:
        logger.error(f"Failed to download weights from HF: {e}")
        return None


# ── LRU model cache ──────────────────────────────────────────

_model_cache: OrderedDict[str, ClassifierModel] = OrderedDict()


def get_model(model_type: str = "resnet18", model_id: str | None = None) -> ClassifierModel:
    """Load and cache model, downloading weights from HF if needed."""
    if model_id:
        result = _resolve_weights(model_id)
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
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning(f"No weights found for {model_type}, using ImageNet pretrained")

    model.eval()
    device = _get_device()
    model = model.to(device)

    if len(_model_cache) >= MAX_CACHED_MODELS:
        evicted_key, evicted = _model_cache.popitem(last=False)
        logger.info(f"Evicted model from cache: {evicted_key}")

    _model_cache[cache_key] = model
    return model


def _find_weights(model_type: str) -> Path | None:
    """Legacy: find best available trained weights for a model type locally."""
    experiments_dir = Path(settings.results_dir)
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
    return best_path


# ── Cache management ─────────────────────────────────────────

_download_progress: float = 0.0
_is_downloading: bool = False
_download_lock = threading.Lock()


def get_cache_status() -> CacheStatus:
    """Get current cache status across all discovered models."""
    models = _discover_models()
    total = len(models)
    cached = sum(1 for m in models if m.is_cached)
    total_size = sum(m.size_mb or 0 for m in models)
    cached_size = sum(m.size_mb or 0 for m in models if m.is_cached)

    with _download_lock:
        progress = _download_progress
        downloading = _is_downloading

    return CacheStatus(
        total_models=total,
        cached_models=cached,
        total_size_mb=total_size,
        cached_size_mb=cached_size,
        is_downloading=downloading,
        download_progress=progress,
        models=models,
    )


def download_all_weights(background: bool = True) -> None:
    """Download all missing model weights from HuggingFace Hub."""
    global _is_downloading, _download_progress

    if background:
        with _download_lock:
            if _is_downloading:
                return
            _is_downloading = True
            _download_progress = 0.0
        thread = threading.Thread(target=_sync_all_weights, daemon=True)
        thread.start()
        return

    _sync_all_weights()


def _sync_all_weights() -> None:
    """Sync all missing weights from HF Hub."""
    global _is_downloading, _download_progress

    try:
        models = _discover_models()
        missing = [m for m in models if not m.is_cached]
        total = len(missing)

        if total == 0:
            logger.info("All models already cached")
            return

        logger.info(f"Downloading {total} model weights from HuggingFace Hub...")

        for i, model in enumerate(missing):
            try:
                _resolve_weights(model.id)
                with _download_lock:
                    _download_progress = (i + 1) / total * 100
                logger.info(f"  [{i+1}/{total}] {model.name} — OK")
            except Exception as e:
                logger.error(f"  [{i+1}/{total}] {model.name} — FAILED: {e}")

        logger.info("Download complete")
    finally:
        with _download_lock:
            _is_downloading = False
            _download_progress = 100.0


def delete_cached_weight(model_id: str) -> bool:
    """Delete a single model's cached weights."""
    model_id = _validate_model_id(model_id)
    experiments_dir = Path(settings.results_dir)
    results_json = experiments_dir / model_id / "results.json"
    if not results_json.exists():
        return False
    results = _safe_json_load(results_json)
    if results is None:
        return False
    best_fold = results.get("best_fold")
    if best_fold is None:
        return False
    pt_name = f"best_fold_{best_fold}.pt"
    cache_pt = _get_cache_dir() / f"{model_id}_{pt_name}"
    if cache_pt.exists():
        cache_pt.unlink()
        return True
    return False


def clear_cache() -> int:
    """Delete ALL cached weights. Returns number of files removed."""
    cache_dir = _get_cache_dir()
    count = 0
    for f in cache_dir.glob("*.pt"):
        f.unlink()
        count += 1
    return count


async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    token = auth[7:]
    payload = decode_jwt(token)
    return {"username": payload["sub"], "role": payload["role"]}


def require_role(min_role: str):
    if min_role == "admin":
        async def _check(user: dict = Depends(get_current_user)):
            if user["role"] != "admin":
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
            return user
        return _check
    else:
        async def _check(user: dict = Depends(get_current_user)):
            return user
        return _check
