# app/core/models/base.py
from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import nn
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Base configuration for all models."""

    num_classes: int = 6
    pretrained: bool = True
    seed: Optional[int] = 42

    def create_model(self) -> "ClassifierModel":
        """Create model instance from this config. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement create_model()")


class ClassifierModel(ABC, nn.Module):
    """
    Abstract base class for all classifier models.

    Any model (ResNet, Siamese, etc.) must implement this interface
    to be used with the training and evaluation pipeline.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self._config = config

    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> "ClassifierModel":
        """
        Factory method to create model from config.

        Args:
            config: Model configuration

        Returns:
            Initialized model instance
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        pass

    @abstractmethod
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from penultimate layer.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Embeddings tensor of shape (batch, embedding_dim)
        """
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """
        Load model weights from file.

        Args:
            path: Path to weights file
        """
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """
        Save model weights to file.

        Args:
            path: Path to save weights
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name for logging and identification."""
        pass

    @property
    def num_classes(self) -> int:
        """Return number of output classes."""
        return self._config.num_classes
