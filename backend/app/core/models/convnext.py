# app/core/models/convnext.py
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

from app.core.models.base import ClassifierModel, ModelConfig

_CONVNEXT_WEIGHTS = {
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),
}

_CONVNEXT_EMBEDDING_DIM = {
    "convnext_tiny": 768,
    "convnext_small": 768,
}


class ConvNeXtConfig(ModelConfig):
    variant: Literal["convnext_tiny", "convnext_small"] = "convnext_tiny"

    def create_model(self) -> "ConvNeXtClassifier":
        return ConvNeXtClassifier(self)


class ConvNeXtClassifier(ClassifierModel):

    def __init__(self, config: ConvNeXtConfig):
        super().__init__(config)
        self._variant = config.variant
        self._embedding_dim = _CONVNEXT_EMBEDDING_DIM[config.variant]
        self._backbone = self._create_backbone(config)

    def _create_backbone(self, config: ConvNeXtConfig) -> nn.Module:
        if config.variant not in _CONVNEXT_WEIGHTS:
            raise ValueError(f"Unknown ConvNeXt variant: {config.variant}")

        model_fn, weights = _CONVNEXT_WEIGHTS[config.variant]
        backbone = model_fn(weights=weights if config.pretrained else None)

        # Replace classifier head: Sequential(LayerNorm2d, Flatten, Linear)
        in_features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Linear(in_features, config.num_classes)

        return backbone

    @classmethod
    def from_config(cls, config: ConvNeXtConfig) -> "ConvNeXtClassifier":
        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = []

        def hook_fn(module, input, output):
            embeddings.append(output.flatten(1))

        # Hook on the LayerNorm2d layer (classifier[0]) — captures features
        # before the final avgpool + flatten + linear
        handle = self._backbone.classifier[0].register_forward_hook(hook_fn)
        try:
            self._backbone(x)
        finally:
            handle.remove()

        return embeddings[0]

    def load_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @property
    def model_name(self) -> str:
        return self._variant

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _unfreeze_head(self) -> None:
        for param in self._backbone.classifier[2].parameters():
            param.requires_grad = True
