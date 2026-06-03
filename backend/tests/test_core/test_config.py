# tests/test_core/test_config.py
from app.config import Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.model_type == "resnet18"
    assert settings.num_classes == 7
    assert settings.epochs == 26
    assert settings.batch_size == 8
    assert settings.learning_rate == 0.001
    assert settings.seed == 42


def test_settings_from_env(monkeypatch):
    """Test settings can be overridden from environment."""
    monkeypatch.setenv("MODEL_TYPE", "resnet50")
    monkeypatch.setenv("EPOCHS", "50")

    settings = Settings()

    assert settings.model_type == "resnet50"
    assert settings.epochs == 50


def test_settings_paths():
    """Test default path settings."""
    settings = Settings()

    assert settings.dataset_dir == "dataset"
    assert settings.models_dir == "models"
    assert settings.results_dir == "experiments"
    assert settings.logs_dir == "logs"


def test_settings_hf_configured():
    """Test HF token detection."""
    settings = Settings()
    assert hasattr(settings, "hf_configured")
    assert isinstance(settings.hf_configured, bool)
