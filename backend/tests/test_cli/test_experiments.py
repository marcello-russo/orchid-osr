# tests/test_cli/test_experiments.py
import torch

from app.cli.definitions import (
    DATASET_PATHS,
    DEFAULT_CONFIG,
    EXPERIMENTS,
)
from app.cli.experiments import parse_args
from app.cli.runner import get_device, show_status


class TestCLI:
    """Tests for CLI module."""

    def test_experiments_defined(self):
        """Test that only valid experiments are defined."""
        assert len(EXPERIMENTS) == 5
        assert all(i in EXPERIMENTS for i in [1, 2, 3, 4])

    def test_experiment_structure(self):
        """Test that experiments have required fields."""
        required_fields = ["name", "train_dataset", "test_dataset", "num_classes", "test_classes", "description"]

        for exp_id, config in EXPERIMENTS.items():
            for field in required_fields:
                assert field in config, f"Exp{exp_id} missing field: {field}"

    def test_dataset_paths_defined(self):
        """Test that required dataset paths are defined."""
        required_datasets = ["clean"]

        for dataset in required_datasets:
            assert dataset in DATASET_PATHS, f"Missing dataset: {dataset}"

    def test_default_config(self):
        """Test default configuration values."""
        assert DEFAULT_CONFIG["model_variant"] == "resnet18"
        assert DEFAULT_CONFIG["epochs"] == 26
        assert DEFAULT_CONFIG["batch_size"] == 8
        assert DEFAULT_CONFIG["learning_rate"] == 0.001
        assert DEFAULT_CONFIG["num_folds"] == 5

    def test_parse_args_default(self, monkeypatch):
        """Test default argument values."""
        monkeypatch.setattr("sys.argv", ["prog"])
        args = parse_args()

        assert args.model == "resnet18"
        assert args.epochs == 26
        assert args.batch_size == 8
        assert args.lr == 0.001
        assert args.folds == 5

    def test_parse_args_model_resnet50(self, monkeypatch):
        """Test model argument with resnet50."""
        monkeypatch.setattr("sys.argv", ["prog", "--model", "resnet50"])
        args = parse_args()

        assert args.model == "resnet50"

    def test_parse_args_epochs_override(self, monkeypatch):
        """Test epochs argument override."""
        monkeypatch.setattr("sys.argv", ["prog", "--epochs", "50"])
        args = parse_args()

        assert args.epochs == 50

    def test_parse_args_folds(self, monkeypatch):
        """Test folds argument for cross-validation."""
        monkeypatch.setattr("sys.argv", ["prog", "--cross-validate", "--folds", "10"])
        args = parse_args()

        assert args.folds == 10
        assert args.cross_validate is True

    def test_parse_args_flags(self, monkeypatch):
        """Test action flags."""
        monkeypatch.setattr("sys.argv", ["prog", "--all", "--train"])
        args = parse_args()

        assert args.all is True
        assert args.train is True
        assert args.test is False

    def test_parse_args_cross_validate(self, monkeypatch):
        """Test cross-validate flag."""
        monkeypatch.setattr("sys.argv", ["prog", "--cross-validate", "--exp", "4"])
        args = parse_args()

        assert args.cross_validate is True
        assert args.exp == 4

    def test_parse_args_exp_selection(self, monkeypatch):
        """Test experiment selection."""
        monkeypatch.setattr("sys.argv", ["prog", "--train", "--exp", "4", "--model", "resnet50"])
        args = parse_args()

        assert args.exp == 4
        assert args.train is True
        assert args.model == "resnet50"

    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert device in [torch.device("cuda"), torch.device("mps"), torch.device("cpu")]

    def test_show_status_runs(self, capsys):
        """Test that show_status runs without error."""
        show_status()
        captured = capsys.readouterr()

        assert "EXPERIMENT STATUS" in captured.out
        assert "Exp6" in captured.out
        assert "Clean split" in captured.out or "clean_split" in captured.out
