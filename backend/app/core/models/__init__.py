# app/core/models/__init__.py
from app.core.models.base import ModelConfig, ClassifierModel
from app.core.models.registry import (
    ModelRegistryEntry,
    MODEL_REGISTRY,
    register_model,
    get_registry_entry,
    create_model,
    create_transform,
    available_models,
)
from app.core.models import resnet as _resnet_mod
from app.core.models import convnext as _convnext_mod
from app.core.models import dinov2 as _dinov2_mod

# Register ResNet variants
register_model("resnet18", ModelRegistryEntry(
    config_cls=_resnet_mod.ResNetConfig,
    classifier_cls=_resnet_mod.ResNetClassifier,
    input_size=(512, 256),
    norm_mean=[0.5364, 0.5518, 0.3866],
    norm_std=[0.2045, 0.2296, 0.2025],
    embedding_dim=512,
))
register_model("resnet50", ModelRegistryEntry(
    config_cls=_resnet_mod.ResNetConfig,
    classifier_cls=_resnet_mod.ResNetClassifier,
    input_size=(512, 256),
    norm_mean=[0.5364, 0.5518, 0.3866],
    norm_std=[0.2045, 0.2296, 0.2025],
    embedding_dim=2048,
))

# Register ConvNeXt variants
register_model("convnext_tiny", ModelRegistryEntry(
    config_cls=_convnext_mod.ConvNeXtConfig,
    classifier_cls=_convnext_mod.ConvNeXtClassifier,
    input_size=(224, 224),
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    embedding_dim=768,
))
register_model("convnext_small", ModelRegistryEntry(
    config_cls=_convnext_mod.ConvNeXtConfig,
    classifier_cls=_convnext_mod.ConvNeXtClassifier,
    input_size=(224, 224),
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    embedding_dim=768,
))

# Register DINOv2 variants (LoRA fine-tuning)
register_model("dinov2_small", ModelRegistryEntry(
    config_cls=_dinov2_mod.DINOv2Config,
    classifier_cls=_dinov2_mod.DINOv2Classifier,
    input_size=(224, 224),
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    embedding_dim=384,
))
register_model("dinov2_base", ModelRegistryEntry(
    config_cls=_dinov2_mod.DINOv2Config,
    classifier_cls=_dinov2_mod.DINOv2Classifier,
    input_size=(224, 224),
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    embedding_dim=768,
))

__all__ = [
    "ModelConfig",
    "ClassifierModel",
    "ResNetConfig",
    "ResNetClassifier",
    "ConvNeXtConfig",
    "ConvNeXtClassifier",
    "DINOv2Config",
    "DINOv2Classifier",
    "ModelRegistryEntry",
    "MODEL_REGISTRY",
    "register_model",
    "get_registry_entry",
    "create_model",
    "create_transform",
    "available_models",
]
