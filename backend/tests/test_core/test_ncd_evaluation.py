import numpy as np

from app.core.ncd.evaluation import compute_ami, compute_f_measure, compute_silhouette


def test_compute_ami_perfect():
    labels = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    ami = compute_ami(labels, pred)
    assert abs(ami - 1.0) < 1e-6


def test_compute_ami_random():
    labels = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([1, 2, 0, 2, 0, 1])
    ami = compute_ami(labels, pred)
    assert -1.0 <= ami <= 1.0


def test_compute_f_measure_perfect():
    labels = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    f = compute_f_measure(labels, pred)
    assert abs(f - 1.0) < 1e-6


def test_compute_f_measure_some_error():
    labels = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 0])
    f = compute_f_measure(labels, pred)
    assert 0.0 <= f <= 1.0


def test_compute_silhouette():
    np.random.seed(57)
    embeddings = np.random.randn(20, 5)
    labels = np.array([0] * 10 + [1] * 10)
    s = compute_silhouette(embeddings, labels)
    assert isinstance(s, float)
    assert -1.0 <= s <= 1.0
