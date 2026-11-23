import json

import pytest

from app.visualization.utils import load_experiment_data


@pytest.fixture
def exp_dir(tmp_path):
    """Create a minimal experiment directory with synthetic data."""
    # config.json
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_variant": "resnet50",
                "num_epochs": 100,
                "batch_size": 8,
                "learning_rate": 0.001,
                "patience": 15,
            }
        )
    )

    # fold_0.json and fold_1.json
    for fold_idx in range(2):
        (tmp_path / f"fold_{fold_idx}.json").write_text(
            json.dumps(
                {
                    "best_f1": 0.88 + fold_idx * 0.01,
                    "best_accuracy": 0.87 + fold_idx * 0.01,
                    "best_epoch": 10 + fold_idx,
                    "final_train_loss": 0.01,
                    "final_val_loss": 0.5,
                    "history": {
                        "train_loss": [1.0, 0.6, 0.3, 0.1, 0.05],
                        "val_loss": [0.9, 0.7, 0.6, 0.55, 0.5],
                        "val_accuracy": [0.5, 0.7, 0.8, 0.85, 0.87],
                        "val_f1": [0.45, 0.65, 0.75, 0.82, 0.88],
                    },
                    "fold": fold_idx,
                    "batch_size": 8,
                }
            )
        )

    # results.json
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "mean_accuracy": 0.875,
                "std_accuracy": 0.01,
                "mean_f1": 0.885,
                "std_f1": 0.01,
                "best_fold": 0,
                "test_metrics": {
                    "accuracy": 0.97,
                    "f1_macro": 0.97,
                    "confusion_matrix": [[10, 1], [2, 15]],
                    "confusion_matrix_labels": ["A", "B"],
                },
                "model_variant": "resnet50",
                "fold_results": [],
            }
        )
    )

    return tmp_path


def test_load_experiment_data_returns_required_keys(exp_dir):
    data = load_experiment_data(str(exp_dir))
    assert "config" in data
    assert "folds" in data
    assert "results" in data


def test_load_experiment_data_folds_count(exp_dir):
    data = load_experiment_data(str(exp_dir))
    assert len(data["folds"]) == 2


def test_load_experiment_data_folds_have_history(exp_dir):
    data = load_experiment_data(str(exp_dir))
    for fold in data["folds"]:
        assert "history" in fold
        assert "train_loss" in fold["history"]
        assert "val_loss" in fold["history"]
        assert "val_f1" in fold["history"]
        assert "val_accuracy" in fold["history"]


def test_load_experiment_data_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_data(str(tmp_path / "nonexistent"))
