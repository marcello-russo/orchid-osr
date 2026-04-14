# app/config.py
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model configuration
    model_type: str = "resnet18"
    num_classes: int = 6

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


# Global settings instance
settings = Settings()
