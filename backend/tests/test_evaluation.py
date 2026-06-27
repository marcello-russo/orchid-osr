import numpy as np
import pytest
from app.core.ncd.evaluation import evaluate_novel, evaluate_loo


def test_novel_auroc_perfect():
    result = evaluate_novel(
        np.array([0.1, 0.2, 0.9, 0.95]),
        np.array([0, 0, 1, 1]),
        novel_name="1",
        class_names=["0", "1"],
    )
    assert result["auroc"] == 1.0
    assert "detection_rate_at_10pct_fpr" in result


def test_novel_auroc_random():
    rng = np.random.default_rng(42)
    result = evaluate_novel(
        rng.uniform(0, 1, 200),
        np.hstack([np.zeros(100), np.ones(100)]),
        novel_name="1",
        class_names=["0", "1"],
    )
    assert 0.4 <= result["auroc"] <= 0.6


def test_novel_nan_on_single_class():
    result = evaluate_novel(
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0, 0, 0, 0]),
        novel_name="1",
        class_names=["0", "1"],
    )
    assert np.isnan(result["auroc"])


def test_novel_class_not_found():
    result = evaluate_novel(
        np.array([0.1, 0.2, 0.3]),
        np.array([0, 0, 0]),
        novel_name="Z",
        class_names=["A", "B"],
    )
    assert np.isnan(result["auroc"])
    assert "error" in result


def test_loo_structure():
    rng = np.random.default_rng(42)
    scores = np.hstack([rng.uniform(0, 0.3, 20), rng.uniform(0.7, 1.0, 40)])
    labels = np.hstack([np.zeros(20), np.ones(20), np.full(20, 2)])
    result = evaluate_loo(scores, labels, class_names=["A", "B", "C"])
    assert "mean_auroc" in result
    assert len(result["per_class"]) == 3
    assert result["per_class"][0]["auroc"] > 0.85
    assert result["per_class"][0]["left_out"] == "A"


def test_loo_handles_missing_class():
    rng = np.random.default_rng(42)
    result = evaluate_loo(
        rng.uniform(0, 1, 40),
        np.hstack([np.zeros(20), np.ones(20)]),
        class_names=["A", "B", "C"],
    )
    assert len(result["per_class"]) == 3
    assert np.isnan(result["per_class"][2]["auroc"])
    assert not np.isnan(result["mean_auroc"])
