# app/core/models/resnet.py
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

from app.core.models.base import ClassifierModel, ModelConfig

_RESNET_WEIGHTS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}

# Multi-grano: Layer 3 GAP + Layer 4 GAP
_MULTI_GRAIN_DIM = {
    "resnet18": 256 + 512,  # layer3=256, layer4=512
    "resnet50": 1024 + 2048,  # layer3=1024, layer4=2048
}


class ResNetConfig(ModelConfig):
    variant: Literal["resnet18", "resnet50"] = "resnet18"
    multi_grain: bool = True

    def create_model(self) -> "ResNetClassifier":
        return ResNetClassifier(self)


class ResNetClassifier(ClassifierModel):
    def __init__(self, config: ResNetConfig):
        super().__init__(config)
        self._variant = config.variant
        self._multi_grain = config.multi_grain
        self._backbone = self._create_backbone(config)

    def _create_backbone(self, config: ResNetConfig) -> nn.Module:
        if config.variant not in _RESNET_WEIGHTS:
            raise ValueError(f"Unknown ResNet variant: {config.variant}")

        model_fn, weights = _RESNET_WEIGHTS[config.variant]
        backbone = model_fn(weights=weights if config.pretrained else None)

        if self._multi_grain:
            self._embedding_dim = _MULTI_GRAIN_DIM[config.variant]
        elif config.variant == "resnet18":
            self._embedding_dim = 512
        else:
            self._embedding_dim = 2048

        num_features = backbone.fc.in_features
        multi_dim = _MULTI_GRAIN_DIM[config.variant] if self._multi_grain else num_features
        backbone.fc = nn.Linear(multi_dim, config.num_classes)

        return backbone

    @classmethod
    def from_config(cls, config: ResNetConfig) -> "ResNetClassifier":
        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._multi_grain:
            if self._config.normalize_embeddings:
                emb = self.get_embeddings(x)
                return self._backbone.fc(torch.nn.functional.normalize(emb, p=2, dim=1))
            return self._backbone(x)

        x = self._backbone.conv1(x)
        x = self._backbone.bn1(x)
        x = self._backbone.relu(x)
        x = self._backbone.maxpool(x)

        x = self._backbone.layer1(x)
        x = self._backbone.layer2(x)
        x = self._backbone.layer3(x)
        layer3_out = x
        x = self._backbone.layer4(x)
        layer4_out = x

        layer3_gap = layer3_out.mean([-2, -1])
        layer4_gap = layer4_out.mean([-2, -1])
        emb = torch.cat([layer3_gap, layer4_gap], dim=1)
        if self._config.normalize_embeddings:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return self._backbone.fc(emb)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if not self._multi_grain:
            embeddings = []

            def hook_fn(module, input, output):
                embeddings.append(output.flatten(1))

            handle = self._backbone.avgpool.register_forward_hook(hook_fn)
            try:
                self._backbone(x)
            finally:
                handle.remove()
            return embeddings[0]

        x = self._backbone.conv1(x)
        x = self._backbone.bn1(x)
        x = self._backbone.relu(x)
        x = self._backbone.maxpool(x)

        x = self._backbone.layer1(x)
        x = self._backbone.layer2(x)
        x = self._backbone.layer3(x)
        layer3_out = x
        x = self._backbone.layer4(x)
        layer4_out = x

        layer3_gap = layer3_out.mean([-2, -1])
        layer4_gap = layer4_out.mean([-2, -1])
        return torch.cat([layer3_gap, layer4_gap], dim=1)

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
        for param in self._backbone.fc.parameters():
            param.requires_grad = True
