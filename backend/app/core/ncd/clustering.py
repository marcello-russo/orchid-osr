"""Clustering algorithms for Novel Class Discovery."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def ss_kmeans_clustering(
    pool_embeddings: np.ndarray,
    labeled_mask: np.ndarray,
    labeled_labels: np.ndarray | None = None,
    n_known: int | None = None,
    n_novel: int = 1,
    max_iter: int = 100,
    random_state: int = 57,
    n_init: int = 20,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Semi-supervised K-means vincolato (Vaze et al., CVPR 2022 — GCD).

    Per 2 novel Exp5 usa 20 init (Vaze repo: increasing inits can help).


    Protocollo GCD fedele: il pool contiene sia campioni etichettati (classi note,
    train) sia non etichettati (classi note + novel, test).

    - Init: centroidi classi note = medie dei labeled; centroidi novel =
      farthest-first tra i SOLI non etichettati.
    - Iterazione (Lloyd vincolato): i labeled restano assegnati al cluster della
      propria classe; i non etichettati vanno al centroide più vicino; i centroidi
      si aggiornano sulle assegnazioni correnti.

    Robust init: ``n_init`` ripetizioni farthest-first con tie-break casuale
    (seed=``random_state``+init). Si tiene il run con inerzia minima, come da
    nota del repo ``sgvaze/generalized-category-discovery``:
    "SS-k-means can be sensitive to bad initialisation under v2, increasing the
    number of inits can help."

    Ritorna (labels, info); labels ha lunghezza = pool e vale 0..K-1 (cluster),
    con i labeled fissati alla propria classe (0..n_known-1).

    Args:
        pool_embeddings: (N, D) array pool (labeled + unlabeled).
        labeled_mask: (N,) bool — True per i campioni labeled.
        labeled_labels: (N_labeled,) o (N,) etichette classi note 0..n_known-1.
            Accetta anche kwarg alias ``labeled_lab`` per compatibilità.
        n_known: numero classi note.
        n_novel: numero classi novel da scoprire.
        max_iter: max iterazioni Lloyd.
        random_state: seed base; init i usa seed=random_state+i.
        n_init: numero di init farthest-first (default 10).

    Reference:
        https://github.com/sgvaze/generalized-category-discovery
    """
    # backward-compat alias: labeled_lab -> labeled_labels
    if labeled_labels is None and "labeled_lab" in kwargs:
        labeled_labels = kwargs.pop("labeled_lab")
    if labeled_labels is None:
        raise TypeError("ss_kmeans_clustering missing required argument: 'labeled_labels' (alias 'labeled_lab')")
    if n_known is None and "n_known" in kwargs:
        n_known = kwargs.pop("n_known")
    if n_known is None:
        raise TypeError("ss_kmeans_clustering missing required argument: 'n_known'")
    # alias n_known/n_novel if passed via kwargs with different names not needed
    if "n_init" in kwargs:
        n_init = kwargs.pop("n_init")
    if "random_state" in kwargs:
        random_state = kwargs.pop("random_state")
    if n_init is not None and n_init < 1:
        raise ValueError("n_init must be >=1")
    if n_novel is not None and n_novel < 0:
        raise ValueError("n_novel must be >=0")
    if n_known is not None and n_known < 1:
        raise ValueError("n_known must be >=1")
    if kwargs:
        raise TypeError(f"unexpected kwargs {list(kwargs)}")

    pool_embeddings = np.asarray(pool_embeddings, dtype=np.float64)
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    labeled_labels = np.asarray(labeled_labels)
    # labeled_labels may be full-length (N,) with dummy for unlabeled, or compact (N_labeled,)
    if labeled_labels.shape[0] == pool_embeddings.shape[0]:
        labeled_labels = labeled_labels[labeled_mask]
    n_total = len(pool_embeddings)
    total_k = n_known + n_novel
    dim = pool_embeddings.shape[1]
    logger.info(
        "SS-KMeans (GCD, vincolato): k=%d (%d note + %d novel), pool=%d (%d labeled, %d unlabeled), n_init=%d",
        total_k,
        n_known,
        n_novel,
        n_total,
        int(labeled_mask.sum()),
        int((~labeled_mask).sum()),
        n_init,
    )

    unlabeled_idx = np.where(~labeled_mask)[0]
    if len(unlabeled_idx) == 0:
        if n_novel > 0:
            raise ValueError("n_novel>0 but no unlabeled samples (labeled_mask all True)")
        # n_novel==0: k=n_known, skip farthest-first, labels = known assignment only
    labeled_pool = pool_embeddings[labeled_mask]

    # Precompute known centroids (fixed across inits)
    known_centroids = np.zeros((n_known, dim))
    for c in range(n_known):
        m = labeled_labels == c
        if not m.any():
            raise ValueError(f"Classe labeled {c} senza campioni")
        known_centroids[c] = labeled_pool[m].mean(axis=0)

    best_labels = None
    best_centroids = None
    best_inertia = float("inf")
    best_n_iter = 0
    best_init = -1

    for init_idx in range(n_init):
        rng = np.random.default_rng(random_state + init_idx)

        # ---- Init centroidi (farthest-first con tie-break casuale) ----
        centroids = np.zeros((total_k, dim))
        centroids[:n_known] = known_centroids
        for j in range(n_novel):
            # distance to nearest already placed centroid
            d = np.linalg.norm(pool_embeddings[unlabeled_idx] - centroids[0], axis=1)
            for c in range(1, n_known + j):
                d = np.minimum(d, np.linalg.norm(pool_embeddings[unlabeled_idx] - centroids[c], axis=1))
            max_d = float(np.max(d))
            # tie-break casuale tra i massimi (tolleranza numerica)
            candidates = np.where(np.isclose(d, max_d, atol=1e-12))[0]
            if len(candidates) > 1:
                pick = int(rng.choice(candidates))
            else:
                pick = int(np.argmax(d))
            centroids[n_known + j] = pool_embeddings[unlabeled_idx][pick]

        # ---- Lloyd vincolato ----
        labels = np.full(n_total, -1, dtype=np.int64)
        # labeled fixed
        # need to map labeled_labels back to positions
        labels[labeled_mask] = labeled_labels
        prev_unlab = None
        n_iter = 0
        for it in range(max_iter):
            n_iter = it + 1
            dmat = np.linalg.norm(pool_embeddings[unlabeled_idx][:, None, :] - centroids[None, :, :], axis=2)
            unlab_assign = np.argmin(dmat, axis=1)
            labels[unlabeled_idx] = unlab_assign
            if prev_unlab is not None and np.array_equal(unlab_assign, prev_unlab):
                break
            prev_unlab = unlab_assign.copy()
            for c in range(total_k):
                member = pool_embeddings[labels == c]
                if len(member):
                    centroids[c] = member.mean(axis=0)

        # inertia: sum squared distance to assigned centroid over whole pool
        dists = np.linalg.norm(pool_embeddings - centroids[labels], axis=1)
        inertia = float(np.sum(dists**2))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()
            best_n_iter = int(n_iter)
            best_init = int(init_idx)

    assert best_labels is not None and best_centroids is not None
    info = {
        "n_iter": best_n_iter,
        "method": "ss_kmeans_constrained_vaze2022",
        "n_known": n_known,
        "n_novel": n_novel,
        "cluster_sizes": [int((best_labels == i).sum()) for i in range(total_k)],
        "inertia": float(best_inertia),
        "n_init": int(n_init),
        "best_init": int(best_init),
    }
    return best_labels, info


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

    from app.core.ncd.evaluation import compute_all_metrics

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
        logger.info(f"  k={k}: silhouette={sil:.4f}  ACC={sup_metrics['acc']:.4f}  NMI={sup_metrics['nmi']:.4f}")
        all_results.append({**sup_metrics, "k": k, "silhouette": sil})
        if sil > best["silhouette"]:
            best = {"k": k, "labels": pred, "silhouette": sil, **sup_metrics}

    return best
