import os
import tempfile

import numpy as np

from app.core.ncd.reporting import (
    plot_auroc_per_class,
    plot_clustering_comparison,
    plot_oscr_curve,
    plot_tsne_known_vs_unknown,
)


def test_plot_tsne_known_vs_unknown_creates_file():
    np.random.seed(57)
    embeddings = np.random.randn(30, 10)
    labels = np.array([0]*15 + [1]*15)
    class_names = ["Class A", "Class B"]
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_tsne_known_vs_unknown(embeddings, labels, class_names, output_dir=tmp)
        assert os.path.exists(path)
        assert path.endswith(".png")


def test_plot_oscr_curve_creates_file():
    curve = [
        {"reject_rate": 0.0, "known_acc": 1.0},
        {"reject_rate": 0.5, "known_acc": 0.7},
        {"reject_rate": 1.0, "known_acc": 0.0},
    ]
    baseline_curve = [
        {"reject_rate": 0.0, "known_acc": 1.0},
        {"reject_rate": 0.5, "known_acc": 0.6},
        {"reject_rate": 1.0, "known_acc": 0.0},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_oscr_curve(curve, baseline_curve, "TestModel", output_dir=tmp)
        assert os.path.exists(path)


def test_plot_auroc_per_class_creates_file():
    per_class = [
        {"left_out": "A", "auroc": 0.9},
        {"left_out": "B", "auroc": 0.8},
        {"left_out": "C", "auroc": 0.85},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_auroc_per_class(per_class, output_dir=tmp)
        assert os.path.exists(path)


def test_plot_clustering_comparison_creates_file():
    data = {
        "kmeans": {"acc": 0.8, "nmi": 0.7, "ari": 0.6, "ami": 0.55, "f_measure": 0.75},
        "dbscan": {"acc": 0.75, "nmi": 0.65, "ari": 0.55, "ami": 0.50, "f_measure": 0.70},
        "spectral": {"acc": 0.78, "nmi": 0.68, "ari": 0.58, "ami": 0.52, "f_measure": 0.72},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_clustering_comparison(data, output_dir=tmp)
        assert os.path.exists(path)
