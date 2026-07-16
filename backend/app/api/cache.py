"""LRU model cache: get_model, _model_cache."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from app.api.weights import _find_weights, _resolve_weights, get_device
from app.config import settings

if TYPE_CHECKING:
    from app.core.models import ClassifierModel

logger = logging.getLogger(__name__)

MAX_CACHED_MODELS = 2
_model_cache: OrderedDict[str, ClassifierModel] = OrderedDict()
_cache_lock = threading.Lock()


def get_model(model_type: str = "resnet18", model_id: str | None = None):
    """Load and cache model, downloading weights from HF if needed."""
    from app.core.models import available_models, create_model

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

    with _cache_lock:
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
    device = get_device()
    model = model.to(device)

    with _cache_lock:
        if len(_model_cache) >= MAX_CACHED_MODELS:
            evicted_key, evicted = _model_cache.popitem(last=False)
            logger.info(f"Evicted model from cache: {evicted_key}")
        _model_cache[cache_key] = model
    return model
