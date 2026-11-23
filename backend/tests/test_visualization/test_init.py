import json
import os

import pytest

from app.visualization import plot_experiment


@pytest.fixture
def full_exp_dir(tmp_path):
    """Complete experiment directory for integration test."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_variant": "resnet50",
                "num_epochs": 10,
                "patience": 5,
            }
        )
    )
    for i in range(2):
        (tmp_path / f"fold_{i}.json").write_text(
            json.dumps(
                {
                    "best_f1": 0.85,
                    "best_accuracy": 0.86,
                    "best_epoch": 3,
                    "final_train_loss": 0.1,
                    "final_val_loss": 0.5,
                    "history": {
                        "train_loss": [1.0, 0.5, 0.3, 0.15, 0.1],
                        "val_loss": [0.9, 0.6, 0.5, 0.48, 0.5],
                        "val_accuracy": [0.5, 0.7, 0.8, 0.85, 0.84],
                        "val_f1": [0.45, 0.65, 0.78, 0.85, 0.83],
                    },
                    "fold": i + 1,
                    "batch_size": 8,
                }
            )
        )
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "mean_accuracy": 0.855,
                "std_accuracy": 0.01,
                "mean_f1": 0.85,
                "std_f1": 0.01,
                "best_fold": 0,
                "test_metrics": {
                    "accuracy": 0.97,
                    "f1_macro": 0.97,
                    "confusion_matrix": [[10, 1], [2, 15]],
                    "confusion_matrix_labels": ["A", "B"],
                },
                "fold_results": [],
            }
        )
    )
    return tmp_path


def test_plot_experiment_creates_all_figures(full_exp_dir):
    output_dir = str(full_exp_dir / "figures")
    plot_experiment(str(full_exp_dir), output_dir)
    expected = [
        "loss_curves.png",
        "metric_curves.png",
        "fold_summary.png",
        "confusion_matrix.png",
        "overfitting_proxy.png",
    ]
    for fname in expected:
        path = os.path.join(output_dir, fname)
        assert os.path.exists(path), f"Missing: {fname}"
        assert os.path.getsize(path) > 0, f"Empty: {fname}"


def test_plot_experiment_default_output_dir(full_exp_dir):
    # Default output_dir should be exp_dir/figures/
    plot_experiment(str(full_exp_dir))
    figures_dir = str(full_exp_dir / "figures")
    assert os.path.isdir(figures_dir)
    assert os.path.exists(os.path.join(figures_dir, "loss_curves.png"))
