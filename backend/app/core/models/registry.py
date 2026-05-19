# app/core/models/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from torchvision import transforms as T

from app.core.models.base import ClassifierModel, ModelConfig


@dataclass
class ModelRegistryEntry:
    config_cls: Type[ModelConfig]
    classifier_cls: Type[ClassifierModel]
    input_size: tuple[int, int]  # (width, height)
    norm_mean: list[float]
    norm_std: list[float]
    embedding_dim: int
    # Architecture-specific training hyperparameters
    optimizer: str = "adam"
    weight_decay: float = 0.0
    learning_rate: float | None = None  # None = use CLI default


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {}


def register_model(name: str, entry: ModelRegistryEntry) -> None:
    MODEL_REGISTRY[name] = entry


def get_registry_entry(model_type: str) -> ModelRegistryEntry:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]


def create_model(
    model_type: str,
    num_classes: int = 6,
    pretrained: bool = True,
    **kwargs,
) -> ClassifierModel:
    entry = get_registry_entry(model_type)
    kwargs.setdefault("variant", model_type)
    config = entry.config_cls(
        num_classes=num_classes, pretrained=pretrained, **kwargs
    )
    return config.create_model()


def create_transform(model_type: str) -> T.Compose:
    from app.preprocessing.resizing import ResizeImageTransform
    entry = get_registry_entry(model_type)
    w, h = entry.input_size
    return T.Compose([
        ResizeImageTransform(w, h),
        T.ToTensor(),
        T.Normalize(mean=entry.norm_mean, std=entry.norm_std),
    ])


def available_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def unique_input_sizes() -> dict[str, tuple[int, int]]:
    """Return unique input sizes across all registered models.

    Returns:
        Dict mapping label "WxH" → (width, height).
    """
    seen = {}
    for entry in MODEL_REGISTRY.values():
        w, h = entry.input_size
        label = f"{w}x{h}"
        if label not in seen:
            seen[label] = (w, h)
    return seen
