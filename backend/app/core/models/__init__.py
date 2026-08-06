# app/core/models/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.models.registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from app.core.models.base import ClassifierModel, ModelConfig  # noqa: F401

__REGISTERED = False


def _register_all():
    """Lazy model registration — avoids importing torch at module load time."""
    global __REGISTERED
    if __REGISTERED:
        return

    from app.core.models import convnext as _convnext_mod
    from app.core.models import dinov2 as _dinov2_mod
    from app.core.models import resnet as _resnet_mod
    from app.core.models.base import ClassifierModel as _ClassifierModel
    from app.core.models.base import ModelConfig as _ModelConfig
    from app.core.models.registry import (
        ModelRegistryEntry,
        register_model,
    )

    globals()["ClassifierModel"] = _ClassifierModel
    globals()["ModelConfig"] = _ModelConfig
    globals()["ModelRegistryEntry"] = ModelRegistryEntry

    register_model(
        "resnet18",
        ModelRegistryEntry(
            config_cls=_resnet_mod.ResNetConfig,
            classifier_cls=_resnet_mod.ResNetClassifier,
            input_size=(256, 512),
            norm_mean=[0.5364, 0.5518, 0.3866],
            norm_std=[0.2045, 0.2296, 0.2025],
            embedding_dim=512,
            optimizer="adam",
            weight_decay=0.0,
            learning_rate=0.001,
        ),
    )
    register_model(
        "resnet50",
        ModelRegistryEntry(
            config_cls=_resnet_mod.ResNetConfig,
            classifier_cls=_resnet_mod.ResNetClassifier,
            input_size=(256, 512),
            norm_mean=[0.5364, 0.5518, 0.3866],
            norm_std=[0.2045, 0.2296, 0.2025],
            embedding_dim=2048,
            optimizer="adam",
            weight_decay=0.0,
            learning_rate=0.001,
        ),
    )

    register_model(
        "convnext_tiny",
        ModelRegistryEntry(
            config_cls=_convnext_mod.ConvNeXtConfig,
            classifier_cls=_convnext_mod.ConvNeXtClassifier,
            input_size=(224, 224),
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            embedding_dim=768,
            optimizer="adamw",
            weight_decay=0.05,
            learning_rate=0.0001,
        ),
    )
    register_model(
        "convnext_small",
        ModelRegistryEntry(
            config_cls=_convnext_mod.ConvNeXtConfig,
            classifier_cls=_convnext_mod.ConvNeXtClassifier,
            input_size=(224, 224),
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            embedding_dim=768,
            optimizer="adamw",
            weight_decay=0.05,
            learning_rate=0.0001,
        ),
    )

    register_model(
        "dinov2_small",
        ModelRegistryEntry(
            config_cls=_dinov2_mod.DINOv2Config,
            classifier_cls=_dinov2_mod.DINOv2Classifier,
            input_size=(224, 224),
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            embedding_dim=384,
            optimizer="adamw",
            weight_decay=0.01,
            learning_rate=0.0002,
        ),
    )
    register_model(
        "dinov2_base",
        ModelRegistryEntry(
            config_cls=_dinov2_mod.DINOv2Config,
            classifier_cls=_dinov2_mod.DINOv2Classifier,
            input_size=(224, 224),
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            embedding_dim=768,
            optimizer="adamw",
            weight_decay=0.01,
            learning_rate=0.0002,
        ),
    )
    __REGISTERED = True


def available_models():
    _register_all()
    return list(MODEL_REGISTRY.keys())


def get_registry_entry(model_type: str):
    _register_all()
    from app.core.models.registry import get_registry_entry as _get

    return _get(model_type)


def create_model(model_type: str, num_classes: int = 6, pretrained: bool = True, **kwargs):
    _register_all()
    from app.core.models.registry import create_model as _create

    return _create(model_type, num_classes=num_classes, pretrained=pretrained, **kwargs)


def create_transform(model_type: str):
    _register_all()
    from app.core.models.registry import create_transform as _create_t

    return _create_t(model_type)


__all__ = [
    "available_models",
    "create_model",
    "create_transform",
    "get_registry_entry",
    "MODEL_REGISTRY",
]
