# app/core/models/registry.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from torchvision import transforms as T

from app.core.models.base import ModelConfig, ClassifierModel


@dataclass
class ModelRegistryEntry:
    config_cls: Type[ModelConfig]
    classifier_cls: Type[ClassifierModel]
    input_size: tuple[int, int]  # (height, width)
    norm_mean: list[float]
    norm_std: list[float]
    embedding_dim: int


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
    entry = get_registry_entry(model_type)
    h, w = entry.input_size
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=entry.norm_mean, std=entry.norm_std),
    ])


def available_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())
