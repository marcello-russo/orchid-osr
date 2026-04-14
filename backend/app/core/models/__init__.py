# app/core/models/__init__.py
from app.core.models.base import ModelConfig, ClassifierModel
from app.core.models.resnet import ResNetConfig, ResNetClassifier

__all__ = [
    "ModelConfig",
    "ClassifierModel",
    "ResNetConfig",
    "ResNetClassifier",
]
