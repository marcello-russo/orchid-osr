import numpy as np

from app.core.ncd.clustering import grid_search_kmeans


def test_grid_search_returns_silhouette():
    np.random.seed(57)
    embeddings = np.random.randn(50, 10)
    labels = np.array([0] * 25 + [1] * 25)
    result = grid_search_kmeans(embeddings, labels, k_range=[2, 3, 4])
    assert "silhouette" in result
    assert result["silhouette"] >= -1.0
    assert result["acc"] >= 0.0
    assert result["k"] in [2, 3, 4]


def test_grid_search_k_matches_range():
    np.random.seed(57)
    embeddings = np.random.randn(50, 10)
    labels = np.array([0] * 25 + [1] * 25)
    result = grid_search_kmeans(embeddings, labels, k_range=[2])
    assert result["k"] == 2
