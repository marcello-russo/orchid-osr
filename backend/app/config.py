# app/config.py
import logging
import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MASTER_SEED: int = 57


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Single source of truth for class names, dataset config, and paths.
    Architecture-specific params (input_size, optimizer, etc.) live in
    the model registry (app/core/models/registry.py).
    """

    # Model configuration
    model_type: str = "resnet18"
    num_classes: int = 6  # 6 training species; the novel class (garganica in Exp 4) is only used for OSR evaluation
    class_names: list[str] = [
        "O. exaltata",
        "O. garganica",
        "O. incubacea",
        "O. majellensis",
        "O. sphegodes",
        "O. sphegodes_Palena",
    ]
    # Novel class di riferimento (Exp 4): O. garganica è esclusa dal training
    # e riservata al test come classe nascosta. Exp 10 usa invece la
    # sphegodes_Napoli legacy, gestita localmente nella sua config.
    novel_class_name: str = "O. garganica"

    # Classes that have internet-sourced images (test set comes from internet)
    internet_classes: list[str] = [
        "O. exaltata",
        "O. garganica",
        "O. incubacea",
        "O. sphegodes",
    ]

    # Clean split config
    test_per_class: int = 50
    split_seed: int = MASTER_SEED

    # Normalization (dataset-specific, used by ResNet variants)
    norm_mean: str = "0.5364 0.5518 0.3866"
    norm_std: str = "0.2045 0.2296 0.2025"

    # Training configuration
    epochs: int = 26
    batch_size: int = 8
    learning_rate: float = 0.001
    seed: int = MASTER_SEED
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
    experiment_comparison_prefix: str = "exp1_ce"

    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_reload: bool = False

    # HuggingFace token (from .env)
    hf_token: str = ""

    # HuggingFace repository URLs (from .env, fallback a default)
    hf_dataset_repo: str = "marcellorusso/orchid-osr-dataset"
    hf_models_repo: str = "marcellorusso/orchid-osr-models"

    # JWT authentication (from .env)
    jwt_secret: str = ""
    jwt_secret_path: str = "data/jwt_secret.txt"

    @model_validator(mode="after")
    def _validate_jwt_secret(self):
        if self.jwt_secret:
            return self

        import secrets

        # Resolve secret path: env var override, then default
        env_path = os.environ.get("JWT_SECRET_PATH", "")
        if env_path:
            secret_file = Path(env_path)
        else:
            secret_file = Path(__file__).resolve().parent.parent / self.jwt_secret_path

        try:
            if secret_file.is_file():
                self.jwt_secret = secret_file.read_text().strip()
                if not self.jwt_secret:
                    raise ValueError("empty secret file")
                logger.info("JWT_SECRET loaded from %s", secret_file)
                return self
        except OSError:
            pass

        # Generate and persist
        self.jwt_secret = secrets.token_hex(32)
        try:
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            secret_file.write_text(self.jwt_secret)
            secret_file.chmod(0o600)
            logger.info("JWT_SECRET generated and saved to %s (mode 600)", secret_file)
        except OSError:
            logger.warning("Could not persist JWT_SECRET to %s", secret_file)
        return self

    superadmin_username: str = ""
    superadmin_password: str = ""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_norm_mean(self) -> list[float]:
        return [float(x) for x in self.norm_mean.split()]

    def get_norm_std(self) -> list[float]:
        return [float(x) for x in self.norm_std.split()]

    @property
    def private_only_classes(self) -> list[str]:
        return [c for c in self.class_names if c not in self.internet_classes]

    @property
    def hf_configured(self) -> bool:
        return bool(self.hf_token) and self.hf_token.startswith("hf_")

    def _check_env_exists(self):
        """Warn if the .env file is missing."""
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if not env_path.exists():
            import warnings

            warnings.warn(f".env not found at {env_path}. Create it from .env.example with your HF_TOKEN.")

    def model_post_init(self, __context):
        self._check_env_exists()
        return super().model_post_init(__context) if hasattr(super(), "model_post_init") else None


# Global settings instance
settings = Settings()
