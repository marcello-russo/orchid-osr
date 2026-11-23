import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from app.core.ncd.delong import delong_auc_ci, delong_paired_test


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    out = delong_auc_ci(y, s)
    assert out["auc"] == 1.0
    assert out["ci_low"] == 1.0 and out["ci_high"] == 1.0


def test_matches_sklearn_and_ci_contains_auc():
    rng = np.random.default_rng(57)
    y = np.array([0] * 60 + [1] * 40)
    s = np.concatenate([rng.normal(0, 1, 60), rng.normal(1, 1, 40)])
    out = delong_auc_ci(y, s)
    assert out["auc"] == pytest.approx(roc_auc_score(y, s))
    assert 0.0 <= out["ci_low"] <= out["auc"] <= out["ci_high"] <= 1.0
    assert out["n_pos"] == 40 and out["n_neg"] == 60


def test_paired_identical_scores():
    rng = np.random.default_rng(7)
    y = np.array([0] * 50 + [1] * 50)
    s = rng.normal(0, 1, 100)
    out = delong_paired_test(y, s, s.copy())
    assert out["diff"] == 0.0
    assert out["p_value"] == 1.0


def test_paired_clearly_better_model():
    rng = np.random.default_rng(11)
    y = np.array([0] * 100 + [1] * 100)
    s_good = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    s_bad = rng.normal(0, 1, 200)
    out = delong_paired_test(y, s_good, s_bad)
    assert out["auc_a"] > out["auc_b"]
    assert 0.0 <= out["p_value"] <= 0.05


def test_all_tied_scores():
    import math

    y = np.array([0, 0, 1, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    out = delong_auc_ci(y, s)
    assert out["auc"] == 0.5
    assert not math.isnan(out["ci_low"]) and not math.isnan(out["ci_high"])
    paired = delong_paired_test(y, s, s.copy())
    assert paired["diff"] == 0.0
    assert paired["p_value"] == 1.0 and not math.isnan(paired["p_value"])
