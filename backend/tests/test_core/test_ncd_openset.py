import numpy as np

from app.core.ncd.openset import (
    baseline_softmax_scores,
    compute_oscr_curve,
    compute_oscr_from_scores,
)


def test_baseline_softmax_scores_shape():
    np.random.seed(57)
    embeddings = np.random.randn(10, 4)
    prototypes = np.random.randn(3, 4)
    prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
    scores = baseline_softmax_scores(embeddings, prototypes)
    assert scores.shape == (10, 4)


def test_baseline_softmax_scores_range():
    np.random.seed(57)
    embeddings = np.random.randn(5, 4)
    prototypes = np.random.randn(2, 4)
    prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
    scores = baseline_softmax_scores(embeddings, prototypes, temperature=1.0)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    np.testing.assert_array_almost_equal(scores[:, :2].sum(axis=1), np.ones(5))


def test_oscr_curve():
    reject_rates = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    known_accs = np.array([1.0, 0.9, 0.7, 0.4, 0.0])
    curve = compute_oscr_curve(reject_rates, known_accs)
    assert len(curve) == 5
    assert curve[0] == {"reject_rate": 0.0, "known_acc": 1.0}
    assert curve[-1] == {"reject_rate": 1.0, "known_acc": 0.0}


def test_oscr_from_scores():
    np.random.seed(57)
    scores = np.random.dirichlet(np.ones(4), size=20)
    # 3 known classes + 1 unknown column
    true_labels = np.array([0, 1, 2, -1]*5)
    curve = compute_oscr_from_scores(scores, true_labels, known_classes=[0, 1, 2], n_thresholds=10)
    assert len(curve) == 10
    for pt in curve:
        assert "reject_rate" in pt
        assert "known_acc" in pt
