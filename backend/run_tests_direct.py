"""Run tests without pytest (that hangs with asyncio/strict config)."""

import sys
import time

sys.path.insert(0, ".")

PASS = 0
FAIL = 0
ERRORS = []


def test(name, func):
    global PASS, FAIL
    try:
        func()
        PASS += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAIL += 1
        ERRORS.append((name, str(e)))
        print(f"  ❌ {name}: {e}")


# === test_evaluation ===
import numpy as np
from app.core.ncd.evaluation import evaluate_novel, evaluate_loo

print("\n=== test_evaluation.py ===")

def test_novel_auroc_perfect():
    r = evaluate_novel(np.array([0.1, 0.2, 0.9, 0.95]), np.array([0, 0, 1, 1]), "1", ["0", "1"])
    assert r["auroc"] == 1.0
    assert "detection_rate_at_10pct_fpr" in r

def test_novel_auroc_random():
    rng = np.random.default_rng(42)
    r = evaluate_novel(rng.uniform(0, 1, 200), np.hstack([np.zeros(100), np.ones(100)]), "1", ["0", "1"])
    assert 0.4 <= r["auroc"] <= 0.6

def test_novel_nan_on_single_class():
    r = evaluate_novel(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]), "1", ["0", "1"])
    assert np.isnan(r["auroc"])

def test_novel_class_not_found():
    r = evaluate_novel(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]), "Z", ["A", "B"])
    assert np.isnan(r["auroc"]) and "error" in r

def test_loo_structure():
    rng = np.random.default_rng(42)
    scores = np.hstack([rng.uniform(0.7, 1.0, 20), rng.uniform(0, 0.3, 20), rng.uniform(0, 0.3, 20)])
    labels = np.hstack([np.zeros(20), np.ones(20), np.full(20, 2)])
    r = evaluate_loo(scores, labels, ["A", "B", "C"])
    assert len(r["per_class"]) == 3 and r["per_class"][0]["auroc"] > 0.85

def test_loo_handles_missing_class():
    rng = np.random.default_rng(42)
    r = evaluate_loo(rng.uniform(0, 1, 40), np.hstack([np.zeros(20), np.ones(20)]), ["A", "B", "C"])
    assert np.isnan(r["per_class"][2]["auroc"]) and not np.isnan(r["mean_auroc"])

test("test_novel_auroc_perfect", test_novel_auroc_perfect)
test("test_novel_auroc_random", test_novel_auroc_random)
test("test_novel_nan_on_single_class", test_novel_nan_on_single_class)
test("test_novel_class_not_found", test_novel_class_not_found)
test("test_loo_structure", test_loo_structure)
test("test_loo_handles_missing_class", test_loo_handles_missing_class)

# === test_detector ===
from app.core.ncd.detector import list_detectors, get_detector

print("\n=== test_detector.py ===")

def test_detector_names():
    names = list_detectors()
    for n in ["Mahalanobis", "KNN", "Energy", "Ensemble"]:
        assert n in names

def test_detector_requires_logits():
    assert not get_detector("Mahalanobis").requires_logits
    assert not get_detector("KNN").requires_logits
    assert get_detector("Energy").requires_logits
    assert get_detector("Ensemble").requires_logits

def test_mahalanobis():
    rng = np.random.default_rng(42)
    train = rng.normal(0, 0.3, (100, 8))
    train_lbl = np.zeros(100, dtype=int)
    test = np.vstack([rng.normal(0, 0.3, (10, 8)), rng.normal(3.0, 0.3, (20, 8))])
    test_lbl = np.hstack([np.zeros(10), np.ones(20)])
    det = get_detector("Mahalanobis")
    det.fit(train, train_lbl)
    r = evaluate_novel(det.score(test), test_lbl, "1", ["0", "1"])
    assert r["auroc"] > 0.95

def test_knn():
    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 0.3, (100, 1))
    train = np.hstack([np.cos(angles), np.sin(angles), np.zeros((100, 6))])
    train_lbl = np.zeros(100, dtype=int)
    test_known = np.hstack([np.cos(angles[:10]), np.sin(angles[:10]), np.zeros((10, 6))])
    na = rng.uniform(np.pi - 0.3, np.pi, (20, 1))
    test_novel = np.hstack([np.cos(na), np.sin(na), np.zeros((20, 6))])
    test = np.vstack([test_known, test_novel])
    test_lbl = np.hstack([np.zeros(10), np.ones(20)])
    det = get_detector("KNN")
    det.fit(train, train_lbl)
    r = evaluate_novel(det.score(test), test_lbl, "1", ["0", "1"])
    assert r["auroc"] > 0.95

def test_knn_small():
    rng = np.random.default_rng(42)
    det = get_detector("KNN")
    det.fit(rng.normal(0, 1, (3, 8)), np.zeros(3, dtype=int))
    s = det.score(rng.normal(0, 1, (5, 8)))
    assert len(s) == 5 and not np.isnan(s).any()

def test_energy():
    rng = np.random.default_rng(42)
    train = rng.normal(0, 0.3, (100, 8))
    test = np.vstack([rng.normal(0, 0.3, (10, 8)), rng.normal(3.0, 0.3, (20, 8))])
    test_lbl = np.hstack([np.zeros(10), np.ones(20)])
    logits = np.zeros((30, 2)); logits[:10, 0] = 5.0; logits[:10, 1] = -5.0
    det = get_detector("Energy")
    det.fit(train, np.zeros(100, dtype=int))
    r = evaluate_novel(det.score(test, logits=logits), test_lbl, "1", ["0", "1"])
    assert r["auroc"] > 0.95

def test_energy_fails():
    det = get_detector("Energy")
    try:
        det.score(np.zeros((10, 8)))
        assert False, "should raise"
    except ValueError:
        pass

def test_ensemble():
    rng = np.random.default_rng(42)
    train = rng.normal(0, 0.3, (100, 8))
    test = np.vstack([rng.normal(0, 0.3, (10, 8)), rng.normal(2.0, 0.3, (20, 8))])
    test_lbl = np.hstack([np.zeros(10), np.ones(20)])
    logits = np.zeros((30, 2)); logits[:10, 0] = 5.0; logits[:10, 1] = -5.0
    det = get_detector("Ensemble")
    det.fit(train, np.zeros(100, dtype=int))
    r = evaluate_novel(det.score(test, logits=logits), test_lbl, "1", ["0", "1"])
    assert r["auroc"] > 0.85

test("test_detector_names", test_detector_names)
test("test_detector_requires_logits", test_detector_requires_logits)
test("test_mahalanobis", test_mahalanobis)
test("test_knn", test_knn)
test("test_knn_small", test_knn_small)
test("test_energy", test_energy)
test("test_energy_fails", test_energy_fails)
test("test_ensemble", test_ensemble)

print(f"\n{'='*40}")
print(f"RESULTS: {PASS} passed, {FAIL} failed")
if ERRORS:
    print("\nErrors:")
    for name, err in ERRORS:
        print(f"  {name}: {err}")
