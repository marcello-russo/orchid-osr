# tests/test_core/test_config.py
from app.config import Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.model_type == "resnet18"
    assert settings.num_classes == 6
    assert settings.novel_class_name == "O. sphegodes_Napoli"
    assert settings.epochs == 26
    assert settings.batch_size == 8
    assert settings.learning_rate == 0.001
    from app.config import MASTER_SEED

    assert settings.seed == MASTER_SEED


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


def test_hf_dataset_repo_format():
    """Test HF dataset repo URL has valid org/repo format."""
    settings = Settings()
    assert settings.hf_dataset_repo.count("/") == 1
    org, repo = settings.hf_dataset_repo.split("/")
    assert org and repo


def test_hf_models_repo_format():
    """Test HF models repo URL has valid org/repo format."""
    settings = Settings()
    assert settings.hf_models_repo.count("/") == 1
    org, repo = settings.hf_models_repo.split("/")
    assert org and repo


def test_hf_repos_are_different():
    """Test dataset and models repos are not the same (copy-paste guard)."""
    settings = Settings()
    assert settings.hf_dataset_repo != settings.hf_models_repo


def test_hf_repo_from_env(monkeypatch):
    """Test HF repo URL can be overridden from environment."""
    monkeypatch.setenv("HF_DATASET_REPO", "other-org/other-dataset")
    monkeypatch.setenv("HF_MODELS_REPO", "other-org/other-models")
    settings = Settings()
    assert settings.hf_dataset_repo == "other-org/other-dataset"
    assert settings.hf_models_repo == "other-org/other-models"
