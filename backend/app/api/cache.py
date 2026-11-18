"""LRU model cache: get_model, _model_cache, get_osr_stats."""

from __future__ import annotations

import logging
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.neighbors import NearestNeighbors

from app.api.weights import _find_weights, _resolve_weights, get_device
from app.config import settings
from app.core.experiment_paths import hf_exp_dir

if TYPE_CHECKING:
    from app.core.models import ClassifierModel

logger = logging.getLogger(__name__)

MAX_CACHED_MODELS = 2
_model_cache: OrderedDict[str, ClassifierModel] = OrderedDict()
_model_cache_lock = threading.Lock()

_osr_stats_cache: dict[str, dict[str, Any]] = {}
_osr_stats_lock = threading.Lock()

_HF_PREFIX_MAP = {
    "exp1_ce": "exp6_ce",
    "exp2_supcon": "exp7_supcon",
    "exp3_sphor": "exp8_sphor",
    "exp4_sphor_ncd": "exp9_ncd",
    "exp5_sphor_ncd": "exp5_sphor_ncd",
}

_STATS_CACHE_DIR = Path.home() / ".cache" / "orchid-weights" / "osr_stats"


def get_model(model_type: str = "resnet18", model_id: str | None = None) -> ClassifierModel:
    """Load and cache model, downloading weights from HF if needed."""
    from app.core.models import available_models, create_model

    weights_path: Path | None
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

    with _model_cache_lock:
        if cache_key in _model_cache:
            _model_cache.move_to_end(cache_key)
            return _model_cache[cache_key]

    model = create_model(variant, num_classes=settings.num_classes, pretrained=True)
    if weights_path:
        model.load_weights(str(weights_path))
        logger.info("Loaded weights from %s", weights_path)
    else:
        logger.warning("No weights found for %s, using ImageNet pretrained", model_type)

    model.eval()
    device = get_device()
    model = model.to(device)

    with _model_cache_lock:
        if len(_model_cache) >= MAX_CACHED_MODELS:
            evicted_key, evicted = _model_cache.popitem(last=False)
            logger.info("Evicted model from cache: %s", evicted_key)
        _model_cache[cache_key] = model
    return model


def get_osr_stats(model_id: str) -> dict[str, Any] | None:
    """Load OSR calibration stats for a model, with LRU cache.

    Returns dict with keys: class_means, cov_inv, nn (NearestNeighbors),
    threshold_ensemble, maha_mean, maha_std, knn_mean, knn_std.
    Returns None if stats are not available.
    """
    with _osr_stats_lock:
        if model_id in _osr_stats_cache:
            return _osr_stats_cache[model_id]

    base = hf_exp_dir(model_id)
    if base is None:
        return None
    stats_path_in_repo = f"{base}/osr_stats.npz"

    _STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _STATS_CACHE_DIR / f"{model_id}.npz"

    if not local_path.exists():
        try:
            downloaded = hf_hub_download(
                repo_id=settings.hf_models_repo,
                filename=stats_path_in_repo,
                token=settings.hf_token or None,
            )
            shutil.copy2(downloaded, local_path)
            logger.info("Downloaded OSR stats: %s", stats_path_in_repo)
        except Exception as e:
            logger.warning("OSR stats not available for %s: %s", model_id, e)
            return None

    try:
        data = np.load(local_path, allow_pickle=False)

        train_embs = data["train_embeddings"]
        nn = NearestNeighbors(n_neighbors=min(5, len(train_embs)), metric="euclidean", n_jobs=1)
        nn.fit(train_embs)

        stats = {
            "class_means": data["class_means"],
            "cov_inv": data["cov_inv"],
            "nn": nn,
            "threshold_ensemble": float(data["threshold_ensemble"][0]),
            "maha_mean": float(data.get("maha_mean", np.array([0.0]))[0]),
            "maha_std": float(data.get("maha_std", np.array([1.0]))[0]),
            "knn_mean": float(data.get("knn_mean", np.array([0.0]))[0]),
            "knn_std": float(data.get("knn_std", np.array([1.0]))[0]),
        }

        with _osr_stats_lock:
            _osr_stats_cache[model_id] = stats
        return stats
    except Exception as e:
        logger.warning("Failed to load OSR stats for %s: %s", model_id, e)
        return None
