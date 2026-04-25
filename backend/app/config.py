# app/config.py
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

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

    # Image dimensions
    img_width: int = 256
    img_height: int = 512

    # Normalization (balanced dataset 256x512)
    norm_mean: str = "0.5364 0.5518 0.3866"
    norm_std: str = "0.2045 0.2296 0.2025"

    # Class names
    class_names: list[str] = [
        "O. exaltata",
        "O. garganica",
        "O. incubacea",
        "O. majellensis",
        "O. sphegodes",
        "O. sphegodes_Palena",
    ]

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


# Global settings instance
settings = Settings()
