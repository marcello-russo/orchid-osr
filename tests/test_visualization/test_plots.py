import os
import pytest
from app.visualization.utils import load_experiment_data
from app.visualization.plots import plot_loss_curves, plot_metric_curves
from app.visualization.style import setup_style


@pytest.fixture
def sample_data(tmp_path):
    """Create minimal experiment dir and return loaded data."""
    import json
    (tmp_path / "config.json").write_text(json.dumps({
        "model_variant": "resnet50", "num_epochs": 10, "patience": 5,
    }))
    for i in range(2):
        (tmp_path / f"fold_{i}.json").write_text(json.dumps({
            "best_f1": 0.85, "best_accuracy": 0.86,
            "best_epoch": 3,
            "final_train_loss": 0.1, "final_val_loss": 0.5,
            "history": {
                "train_loss": [1.0, 0.5, 0.3, 0.15, 0.1],
                "val_loss": [0.9, 0.6, 0.5, 0.48, 0.5],
                "val_accuracy": [0.5, 0.7, 0.8, 0.85, 0.84],
                "val_f1": [0.45, 0.65, 0.78, 0.85, 0.83],
            },
            "fold": i + 1, "batch_size": 8,
        }))
    (tmp_path / "results.json").write_text(json.dumps({
        "mean_accuracy": 0.855, "std_accuracy": 0.01,
        "mean_f1": 0.85, "std_f1": 0.01, "best_fold": 0,
        "fold_results": [],
    }))
    return load_experiment_data(str(tmp_path))


def test_plot_loss_curves_creates_file(sample_data, tmp_path):
    setup_style()
    output_dir = str(tmp_path / "figures")
    os.makedirs(output_dir, exist_ok=True)
    plot_loss_curves(sample_data, output_dir)
    assert os.path.exists(os.path.join(output_dir, "loss_curves.png"))
    assert os.path.getsize(os.path.join(output_dir, "loss_curves.png")) > 0


def test_plot_metric_curves_creates_file(sample_data, tmp_path):
    setup_style()
    output_dir = str(tmp_path / "figures")
    os.makedirs(output_dir, exist_ok=True)
    plot_metric_curves(sample_data, output_dir)
    assert os.path.exists(os.path.join(output_dir, "metric_curves.png"))
    assert os.path.getsize(os.path.join(output_dir, "metric_curves.png")) > 0
