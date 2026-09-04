"""Model discovery, weight download, and cache management."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.api.schemas import CacheStatus, ModelOption
from app.cli.definitions import EXPERIMENTS as _EXPERIMENTS
from app.config import settings

logger = logging.getLogger(__name__)

CLASS_NAMES = settings.class_names  # 6 training classes
NOVEL_CLASS_NAME = settings.novel_class_name
# Mapping from experiment directory prefix → human label + number
_EXP_META: dict[str, tuple[str, int]] = {
    "exp1_ce": ("Exp 1 (CE)", 1),
    "exp2_supcon": ("Exp 2 (SupCon)", 2),
    "exp3_sphor": ("Exp 3 (SpHOR)", 3),
    "exp4_sphor_ncd": ("Exp 4 (SpHOR-NCD)", 4),
    "exp5_sphor_ncd": ("Exp 5 (SpHOR-NCD 2 novel)", 5),
}

_HF_PREFIX_MAP: dict[str, str] = {
    "exp1_ce": "exp6_ce",
    "exp2_supcon": "exp7_supcon",
    "exp3_sphor": "exp8_sphor",
    "exp4_sphor_ncd": "exp9_ncd",
    "exp5_sphor_ncd": "exp5_sphor_ncd",
}


def _get_exp_num(model_id: str) -> int | None:
    """Extract experiment number from a model_id (e.g. exp3_sphor_resnet18 -> 3)."""
    for prefix, (_, num) in _EXP_META.items():
        if model_id.startswith(prefix):
            return num
    return None


_VARIANT_DISPLAY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "dinov2_small": "DINOv2-Small",
    "dinov2_base": "DINOv2-Base",
}

_download_progress: float = 0.0
_is_downloading: bool = False
_download_lock = threading.Lock()


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _validate_model_id(model_id: str) -> str:
    if ".." in model_id or "/" in model_id or "\\" in model_id:
        raise ValueError(f"Invalid model_id: {model_id}")
    return model_id


# ── Model discovery ──────────────────────────────────────────


def _discover_models() -> list[ModelOption]:
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

        pt_name = f"best_fold_{best_fold}.pt"
        local_pt = cache_dir / f"{dir_name}_{pt_name}"
        is_cached = local_pt.exists()
        if not is_cached:
            orig_pt = parent / pt_name
            is_cached = orig_pt.exists()

        options.append(
            ModelOption(
                id=dir_name,
                name=_format_model_name(dir_name, mv),
                model_variant=mv,
                mean_f1=_get_mean_f1(results),
                std_f1=results.get("std_f1"),
                test_accuracy=test_metrics.get("accuracy"),
                num_folds=results.get("num_folds"),
                is_cached=is_cached,
                size_mb=_estimate_pt_size(dir_name),
                exp_num=exp_num,
                exp_label=exp_label,
            )
        )

    # Classify serving is 6-class only: drop non-6-class heads (their
    # checkpoints would crash load_state_dict and mislabel indices) and
    # models below the quality gate (mean F1 < 0.8). Data-driven from
    # config.json/results.json — no hardcoded experiment names.
    MIN_SERVE_F1 = 0.8
    seen: set[str] = set()
    deduped: list[ModelOption] = []
    for m in options:
        if m.id not in seen:
            seen.add(m.id)
            cfg = _safe_json_load(Path(settings.results_dir) / m.id / "config.json") or {}
            if cfg.get("num_classes", 6) != 6:
                continue
            if (m.mean_f1 or 0) < MIN_SERVE_F1:
                continue
            deduped.append(m)

    # Compute tags on deduped models only
    ncd_exps = {num for num, cfg in _EXPERIMENTS.items() if cfg.get("ncd_mode")}
    for m in deduped:
        if m.exp_num in ncd_exps:
            m.tags.append("ncd")

    best_f1 = max((m.mean_f1 for m in deduped if m.mean_f1 is not None), default=None)
    if best_f1 is not None:
        for m in deduped:
            if m.mean_f1 == best_f1:
                m.tags.append("best_f1")
                break

    deduped.sort(key=lambda m: (-m.exp_num, -(m.mean_f1 or 0)))
    return deduped


def get_model_options() -> list[ModelOption]:
    return _discover_models()


# ── Weight resolution with HF fallback ───────────────────────


def _resolve_weights(model_id: str) -> tuple[Path, str] | None:
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

    local_pt = results_json.parent / pt_name
    if local_pt.exists():
        return local_pt, variant

    cache_pt = _get_cache_dir() / f"{model_id}_{pt_name}"
    if cache_pt.exists():
        return cache_pt, variant

    hf_path = _hf_path_for(model_id, pt_name)
    if hf_path is None:
        logger.error(f"Cannot resolve HF path for {model_id}")
        return None

    try:
        downloaded = hf_hub_download(
            repo_id=settings.hf_models_repo,
            filename=hf_path,
            token=settings.hf_token or None,
        )
        cache_pt = _get_cache_dir() / f"{model_id}_{pt_name}"
        if not cache_pt.exists():
            shutil.copy2(downloaded, cache_pt)
        logger.info(f"Downloaded weights from HF: {hf_path} ({downloaded})")
        return cache_pt, variant
    except Exception as e:
        logger.error(f"Failed to download weights from HF: {e}")
        return None


def _find_weights(model_type: str) -> Path | None:
    """Find the best available weights for a model type.

    Scans all discovered experiments for the given model type and returns
    the path to the best-performing checkpoint (highest test F1).
    Falls back to HF Hub download if not found locally.
    """
    experiments_dir = Path(settings.results_dir)
    best_score = -1.0
    best_path: Path | None = None
    for results_json in experiments_dir.rglob("results.json"):
        results = _safe_json_load(results_json)
        if results is None:
            continue
        mv = results.get("model_variant", "")
        if mv != model_type:
            continue
        parent = results_json.parent
        dir_name = parent.name
        cfg = _safe_json_load(parent / "config.json") or {}
        if cfg.get("num_classes", 6) != 6:
            continue
        best_fold = results.get("best_fold")
        if best_fold is None:
            continue
        candidate = parent / f"best_fold_{best_fold}.pt"
        if candidate.exists():
            score = _get_mean_f1(results) or 0
            if score > best_score:
                best_score = score
                best_path = candidate
        else:
            # Try downloading from HF
            try:
                hf_path = _hf_path_for(dir_name, f"best_fold_{best_fold}.pt")
                if hf_path:
                    downloaded = hf_hub_download(
                        repo_id=settings.hf_models_repo,
                        filename=hf_path,
                        token=settings.hf_token or None,
                    )
                    cache_pt = _get_cache_dir() / f"{dir_name}_best_fold_{best_fold}.pt"
                    if not cache_pt.exists():
                        shutil.copy2(downloaded, cache_pt)
                    score = _get_mean_f1(results) or 0
                    if score > best_score:
                        best_score = score
                        best_path = cache_pt
            except Exception:
                continue
    return best_path


# ── Cache management ─────────────────────────────────────────


def get_cache_status() -> CacheStatus:
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
                logger.info(f"  [{i + 1}/{total}] {model.name} — OK")
            except Exception as e:
                logger.error(f"  [{i + 1}/{total}] {model.name} — FAILED: {e}")

        logger.info("Download complete")
    finally:
        with _download_lock:
            _is_downloading = False
            _download_progress = 100.0


def delete_cached_weight(model_id: str) -> bool:
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
    cache_dir = _get_cache_dir()
    count = 0
    for f in cache_dir.glob("*.pt"):
        f.unlink()
        count += 1
    return count
