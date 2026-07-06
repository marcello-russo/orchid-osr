# app/core/models/convnext.py
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Multi-grano: Stage 3 GAP + Stage 4 GAP
_MULTI_GRAIN_DIM = {
    "convnext_tiny": 384 + 768,    # stage3=384, stage4=768
    "convnext_small": 384 + 768,
}


class ConvNeXtConfig(ModelConfig):
    variant: Literal["convnext_tiny", "convnext_small"] = "convnext_tiny"
    multi_grain: bool = True

    def create_model(self) -> "ConvNeXtClassifier":
        return ConvNeXtClassifier(self)


class ConvNeXtClassifier(ClassifierModel):

    def __init__(self, config: ConvNeXtConfig):
        super().__init__(config)
        self._variant = config.variant
        self._multi_grain = config.multi_grain
        if self._multi_grain:
            self._embedding_dim = _MULTI_GRAIN_DIM[config.variant]
            self._embedding_dim_backup = _CONVNEXT_EMBEDDING_DIM[config.variant]
        else:
            self._embedding_dim = _CONVNEXT_EMBEDDING_DIM[config.variant]
        self._backbone = self._create_backbone(config)

    def _create_backbone(self, config: ConvNeXtConfig) -> nn.Module:
        if config.variant not in _CONVNEXT_WEIGHTS:
            raise ValueError(f"Unknown ConvNeXt variant: {config.variant}")

        model_fn, weights = _CONVNEXT_WEIGHTS[config.variant]
        backbone = model_fn(weights=weights if config.pretrained else None)

        in_features = backbone.classifier[2].in_features
        multi_dim = _MULTI_GRAIN_DIM[config.variant] if self._multi_grain else in_features
        backbone.classifier[2] = nn.Linear(multi_dim, config.num_classes)

        return backbone

    @classmethod
    def from_config(cls, config: ConvNeXtConfig) -> "ConvNeXtClassifier":
        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._multi_grain:
            return self._backbone(x)

        feats = self._backbone.features
        # Forward through stages
        for i, stage in enumerate(feats):
            x = stage(x)
            if i == 4:
                stage3_out = x
        stage4_out = x

        stage3_gap = stage3_out.mean([-2, -1])
        stage4_gap = stage4_out.mean([-2, -1])
        emb = torch.cat([stage3_gap, stage4_gap], dim=1)
        emb = F.layer_norm(emb, [emb.shape[-1]])
        return self._backbone.classifier[2](emb)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if not self._multi_grain:
            embeddings = []
            def hook_fn(module, input, output):
                embeddings.append(output.flatten(1))
            handle = self._backbone.classifier[0].register_forward_hook(hook_fn)
            try:
                self._backbone(x)
            finally:
                handle.remove()
            return embeddings[0]

        feats = self._backbone.features
        for i, stage in enumerate(feats):
            x = stage(x)
            if i == 4:
                stage3_out = x
        stage4_out = x

        stage3_gap = stage3_out.mean([-2, -1])
        stage4_gap = stage4_out.mean([-2, -1])
        emb = torch.cat([stage3_gap, stage4_gap], dim=1)
        return F.layer_norm(emb, [emb.shape[-1]])

    def load_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
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
