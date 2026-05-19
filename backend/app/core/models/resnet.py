# app/core/models/resnet.py
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

from app.core.models.base import ClassifierModel, ModelConfig

# Map variant to weights enum (avoids deprecated pretrained= param)
_RESNET_WEIGHTS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


class ResNetConfig(ModelConfig):
    """Configuration for ResNet models."""

    variant: Literal["resnet18", "resnet50"] = "resnet18"

    def create_model(self) -> "ResNetClassifier":
        """Create ResNetClassifier from this config."""
        return ResNetClassifier(self)


class ResNetClassifier(ClassifierModel):
    """
    ResNet18 or ResNet50 classifier for orchid classification.

    Supports pretrained ImageNet weights and custom number of classes.
    """

    def __init__(self, config: ResNetConfig):
        super().__init__(config)
        self._variant = config.variant
        self._backbone = self._create_backbone(config)

    def _create_backbone(self, config: ResNetConfig) -> nn.Module:
        """Create ResNet backbone with modified classification head."""
        if config.variant not in _RESNET_WEIGHTS:
            raise ValueError(f"Unknown ResNet variant: {config.variant}")

        model_fn, weights = _RESNET_WEIGHTS[config.variant]
        backbone = model_fn(weights=weights if config.pretrained else None)

        if config.variant == "resnet18":
            self._embedding_dim = 512
        else:
            self._embedding_dim = 2048

        # Replace final fully connected layer
        num_features = backbone.fc.in_features
        backbone.fc = nn.Linear(num_features, config.num_classes)

        return backbone

    @classmethod
    def from_config(cls, config: ResNetConfig) -> "ResNetClassifier":
        """Create ResNetClassifier from config."""
        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet.

        Args:
            x: Input images of shape (batch, 3, H, W)

        Returns:
            Logits of shape (batch, num_classes)
        """
        return self._backbone(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from penultimate layer.

        Thread-safe: uses a forward hook to capture the avgpool output
        without mutating the model's FC layer.

        Args:
            x: Input images of shape (batch, 3, H, W)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        embeddings = []

        def hook_fn(module, input, output):
            # avgpool output is (batch, channels, 1, 1) — flatten to (batch, channels)
            embeddings.append(output.flatten(1))

        handle = self._backbone.avgpool.register_forward_hook(hook_fn)
        try:
            self._backbone(x)
        finally:
            handle.remove()

        return embeddings[0]

    def load_weights(self, path: str) -> None:
        """Load weights from file."""
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)

    def save_weights(self, path: str) -> None:
        """Save weights to file."""
        torch.save(self.state_dict(), path)

    @property
    def model_name(self) -> str:
        """Return model variant name."""
        return self._variant

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def _unfreeze_head(self) -> None:
        for param in self._backbone.fc.parameters():
            param.requires_grad = True
