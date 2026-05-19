# app/config.py
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Single source of truth for class names, dataset config, and paths.
    Architecture-specific params (input_size, optimizer, etc.) live in
    the model registry (app/core/models/registry.py).
    """

    # Model configuration
    model_type: str = "resnet18"
    num_classes: int = 6
    class_names: list[str] = [
        "O. exaltata",
        "O. garganica",
        "O. incubacea",
        "O. majellensis",
        "O. sphegodes",
        "O. sphegodes_Palena",
    ]

    # Classes that have internet-sourced images (test set comes from internet)
    internet_classes: list[str] = [
        "O. exaltata",
        "O. garganica",
        "O. incubacea",
        "O. sphegodes",
    ]

    # Clean split config
    test_per_class: int = 50
    split_seed: int = 42

    # Normalization (dataset-specific, used by ResNet variants)
    norm_mean: str = "0.5364 0.5518 0.3866"
    norm_std: str = "0.2045 0.2296 0.2025"

    # Training configuration
    epochs: int = 26
    batch_size: int = 8
    learning_rate: float = 0.001
    seed: int = 42
    patience: int = 0

    # Path configuration
    dataset_dir: str = "dataset"
    models_dir: str = "models"
    results_dir: str = "experiments"
    logs_dir: str = "logs"

    # Cropper configuration
    cropper_model_path: str = "models/cropper/fasterrcnn_orchid3.pth"
    cropper_score_threshold: float = 0.5
    cropper_num_classes: int = 2  # background + flower

    # Comparison (which experiment prefix to compare in the results table)
    experiment_comparison_prefix: str = "exp6_clean_split"

    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_reload: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    def get_norm_mean(self) -> list[float]:
        return [float(x) for x in self.norm_mean.split()]

    def get_norm_std(self) -> list[float]:
        return [float(x) for x in self.norm_std.split()]

    @property
    def private_only_classes(self) -> list[str]:
        return [c for c in self.class_names if c not in self.internet_classes]

    @property
    def hf_configured(self) -> bool:
        token = os.environ.get("HF_TOKEN", "")
        return bool(token) and token.startswith("hf_")


    def _check_env_exists(self):
        """Warn if the .env file is missing."""
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if not env_path.exists():
            import warnings
            warnings.warn(
                f".env not found at {env_path}. Create it from .env.example with your HF_TOKEN."
            )

    def model_post_init(self, __context):
        self._check_env_exists()
        return super().model_post_init(__context) if hasattr(super(), "model_post_init") else None


# Global settings instance
settings = Settings()
