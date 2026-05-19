"""Clustering algorithms for Novel Class Discovery."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 57,
    n_init: int = 10,
) -> tuple[np.ndarray, dict]:
    """K-means clustering."""
    from sklearn.cluster import KMeans
    logger.info(f"K-means: k={n_clusters}, n_init={n_init}")
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(embeddings)
    info = {
        "inertia": float(km.inertia_),
        "n_iter": int(km.n_iter_),
        "cluster_sizes": [int((labels == i).sum()) for i in range(n_clusters)],
    }
    return labels, info


def dbscan_clustering(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> tuple[np.ndarray, dict]:
    """DBSCAN clustering (discovers number of clusters automatically)."""
    from sklearn.cluster import DBSCAN
    logger.info(f"DBSCAN: eps={eps}, min_samples={min_samples}")
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(embeddings)
    n_clusters = len(set(labels[labels >= 0]))
    n_noise = int((labels == -1).sum())
    logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
    info = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "eps": eps,
        "min_samples": min_samples,
    }
    return labels, info


def spectral_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 57,
) -> tuple[np.ndarray, dict]:
    """Spectral clustering."""
    from sklearn.cluster import SpectralClustering
    logger.info(f"Spectral clustering: k={n_clusters}")
    sc = SpectralClustering(
        n_clusters=n_clusters,
        random_state=random_state,
        affinity="nearest_neighbors",
        n_jobs=-1,
    )
    labels = sc.fit_predict(embeddings)
    info = {
        "cluster_sizes": [int((labels == i).sum()) for i in range(n_clusters)],
    }
    return labels, info


def grid_search_kmeans(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    k_range: list[int],
    random_state: int = 57,
) -> dict:
    """Grid search over k for k-means.

    Uses SILHOUETTE score (unsupervised) to select best k.
    ACC and other supervised metrics are computed only for evaluation.
    """
    from sklearn.metrics import silhouette_score

    from app.core.ncd.metrics import compute_all_metrics

    best = {"silhouette": -2.0, "k": None, "labels": None}
    all_results = []

    for k in k_range:
        pred, info = kmeans_clustering(embeddings, k, random_state=random_state)
        n_unique = len(set(pred))
        if n_unique < 2 or n_unique >= len(embeddings):
            sil = -1.0
        else:
            sil = float(silhouette_score(embeddings, pred))
        sup_metrics = compute_all_metrics(true_labels, pred)
        logger.info(
            f"  k={k}: silhouette={sil:.4f}  ACC={sup_metrics['acc']:.4f}"
            f"  NMI={sup_metrics['nmi']:.4f}"
        )
        all_results.append({**sup_metrics, "k": k, "silhouette": sil})
        if sil > best["silhouette"]:
            best = {"k": k, "labels": pred, "silhouette": sil, **sup_metrics}

    return best
