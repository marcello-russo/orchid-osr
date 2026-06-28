import numpy as np
import pytest
from app.core.ncd.detector import list_detectors, get_detector
from app.core.ncd.evaluation import evaluate_novel


def test_detector_names():
    names = list_detectors()
    assert "Mahalanobis" in names
    assert "KNN" in names
    assert "Energy" in names
    assert "Ensemble" in names
    assert len(names) == len(set(names))


def test_detector_requires_logits():
    assert not get_detector("Mahalanobis").requires_logits
    assert not get_detector("KNN").requires_logits
    assert get_detector("Energy").requires_logits
    assert get_detector("Ensemble").requires_logits


def test_mahalanobis_detects_novel():
    rng = np.random.default_rng(42)
    train_emb = rng.normal(0, 0.3, (100, 8))
    train_labels = np.zeros(100, dtype=int)
    test_known = rng.normal(0, 0.3, (10, 8))
    test_novel = rng.normal(3.0, 0.3, (20, 8))
    test_emb = np.vstack([test_known, test_novel])
    test_labels = np.hstack([np.zeros(10, dtype=int), np.ones(20, dtype=int)])

    det = get_detector("Mahalanobis")
    det.fit(train_emb, train_labels)
    scores = det.score(test_emb)
    result = evaluate_novel(scores, test_labels, novel_name="1", class_names=["0", "1"])
    assert result["auroc"] > 0.95


def test_knn_detects_novel():
    rng = np.random.default_rng(42)
    # KNN con L2 normalization: separazione angolare (come embeddings reali)
    angles = rng.uniform(0, 0.3, (100, 1))
    train_emb = np.hstack([np.cos(angles), np.sin(angles), np.zeros((100, 6))])
    train_labels = np.zeros(100, dtype=int)

    test_known = np.hstack([np.cos(angles[:10]), np.sin(angles[:10]), np.zeros((10, 6))])
    novel_angles = rng.uniform(np.pi - 0.3, np.pi, (20, 1))
    test_novel = np.hstack([np.cos(novel_angles), np.sin(novel_angles), np.zeros((20, 6))])

    test_emb = np.vstack([test_known, test_novel])
    test_labels = np.hstack([np.zeros(10), np.ones(20)])

    det = get_detector("KNN")
    det.fit(train_emb, train_labels)
    scores = det.score(test_emb)
    result = evaluate_novel(scores, test_labels, novel_name="1", class_names=["0", "1"])
    assert result["auroc"] > 0.95


def test_knn_handles_small_train():
    rng = np.random.default_rng(42)
    train_emb = rng.normal(0, 1, (3, 8))
    train_labels = np.zeros(3, dtype=int)
    det = get_detector("KNN")
    det.fit(train_emb, train_labels)
    scores = det.score(rng.normal(0, 1, (5, 8)))
    assert len(scores) == 5
    assert not np.isnan(scores).any()


def test_energy_fails_without_logits():
    det = get_detector("Energy")
    with pytest.raises(ValueError, match="requires logits"):
        det.score(np.zeros((10, 8)))


def test_energy_detects_novel():
    rng = np.random.default_rng(42)
    train_emb = rng.normal(0, 0.3, (100, 8))
    train_labels = np.zeros(100, dtype=int)
    test_known = rng.normal(0, 0.3, (10, 8))
    test_novel = rng.normal(3.0, 0.3, (20, 8))
    test_emb = np.vstack([test_known, test_novel])
    test_labels = np.hstack([np.zeros(10), np.ones(20)])
    logits = np.zeros((30, 2))
    logits[:10, 0] = 5.0
    logits[:10, 1] = -5.0
    logits[10:] = 0.0

    det = get_detector("Energy")
    det.fit(train_emb, train_labels)
    scores = det.score(test_emb, logits=logits)
    result = evaluate_novel(scores, test_labels, novel_name="1", class_names=["0", "1"])
    assert result["auroc"] > 0.95


def test_ensemble_detects_novel():
    rng = np.random.default_rng(42)
    train_emb = rng.normal(0, 0.3, (100, 8))
    train_labels = np.zeros(100, dtype=int)
    test_known = rng.normal(0, 0.3, (10, 8))
    test_novel = rng.normal(2.0, 0.3, (20, 8))
    test_emb = np.vstack([test_known, test_novel])
    test_labels = np.hstack([np.zeros(10), np.ones(20)])
    logits = np.zeros((30, 2))
    logits[:10, 0] = 5.0
    logits[:10, 1] = -5.0
    logits[10:] = 0.0

    det = get_detector("Ensemble")
    det.fit(train_emb, train_labels)
    scores = det.score(test_emb, logits=logits)
    result = evaluate_novel(scores, test_labels, novel_name="1", class_names=["0", "1"])
    assert result["auroc"] > 0.85
