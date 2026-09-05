"""Tests for app.core.ncd.comparison (no network, tmp dirs)."""

import json
import os


def _make_ncd_exp(base, name, auroc=0.8):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "ncd_results.json"), "w") as f:
        json.dump(
            {
                "detectors": {"Ensemble": {"auroc": auroc}},
                "clustering": {
                    "kmeans_best": {
                        "acc": 0.7,
                        "nmi": 0.6,
                        "ari": 0.5,
                        "ami": 0.5,
                        "f_measure": 0.6,
                        "silhouette": 0.3,
                    },
                    "selected_k": 6,
                },
            },
            f,
        )
    return d


def test_build_comparison_collects_and_saves(tmp_path):
    from app.core.ncd.comparison import build_comparison

    _make_ncd_exp(str(tmp_path), "exp5_sphor_ncd_resnet18", auroc=0.82)
    entries = build_comparison(str(tmp_path))
    assert len(entries) == 1
    assert entries[0]["model"] == "resnet18"
    assert entries[0]["mean_auroc"] == 0.82
    assert entries[0]["ensemble_auroc"] == 0.82
    assert os.path.exists(os.path.join(str(tmp_path), "ncd_comparison.json"))


def test_build_comparison_empty_dir(tmp_path):
    from app.core.ncd.comparison import build_comparison

    assert build_comparison(str(tmp_path)) == []
