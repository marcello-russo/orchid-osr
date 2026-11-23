"""Clustering algorithms for Novel Class Discovery."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_labeled_mask(n_train: int, n_test: int) -> np.ndarray:
    """Maschera labeled pura da posizione/blocco (mai dai nomi specie).

    I primi ``n_train`` punti del pool sono i labeled (train), i successivi
    ``n_test`` sono gli unlabeled (test). Nessuna label in input per
    costruzione (barriera anti-leakage strutturale).
    """
    mask = np.zeros(n_train + n_test, dtype=bool)
    mask[:n_train] = True
    return mask


def _farthest_first_novel(
    pool_unlabeled: np.ndarray,
    known_centroids: np.ndarray,
    n_novel: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """Init novel farthest-first (estratto verbatim da ss_kmeans, puro).

    Nessuna label in input (barriera anti-leakage strutturale).
    Qualsiasi kwarg (in particolare label-like) è rifiutato con TypeError:
    la firma accetta solo (embedding, centroidi, n_novel, rng).
    Ritorna (n_novel, D) centroidi novel.
    """
    if kwargs:
        raise TypeError(f"_farthest_first_novel rifiuta kwarg spuri {sorted(kwargs)} (anti-leakage: niente label)")
    pool_unlabeled = np.asarray(pool_unlabeled, dtype=np.float64)
    known_centroids = np.asarray(known_centroids, dtype=np.float64)
    n_known = known_centroids.shape[0]
    dim = pool_unlabeled.shape[1]
    centroids = np.zeros((n_known + n_novel, dim))
    centroids[:n_known] = known_centroids
    for j in range(n_novel):
        # distance to nearest already placed centroid
        d = np.linalg.norm(pool_unlabeled - centroids[0], axis=1)
        for c in range(1, n_known + j):
            d = np.minimum(d, np.linalg.norm(pool_unlabeled - centroids[c], axis=1))
        max_d = float(np.max(d))
        # tie-break casuale tra i massimi (tolleranza numerica)
        candidates = np.where(np.isclose(d, max_d, atol=1e-12))[0]
        if len(candidates) > 1:
            pick = int(rng.choice(candidates))
        else:
            pick = int(np.argmax(d))
        centroids[n_known + j] = pool_unlabeled[pick]
    return centroids[n_known:]


def _constrained_lloyd(
    pool: np.ndarray,
    labeled_mask: np.ndarray,
    labeled_labels: np.ndarray,
    centroids_init: np.ndarray,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Un giro di Lloyd vincolato (estratto verbatim da ss_kmeans).

    Ritorna (labels, centroids, inertia, n_iter).
    """
    pool = np.asarray(pool, dtype=np.float64)
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    labeled_labels = np.asarray(labeled_labels)
    if labeled_labels.shape[0] == pool.shape[0]:
        labeled_labels = labeled_labels[labeled_mask]
    n_total = len(pool)
    total_k = centroids_init.shape[0]
    centroids = np.asarray(centroids_init, dtype=np.float64).copy()
    unlabeled_idx = np.where(~labeled_mask)[0]
    labels = np.full(n_total, -1, dtype=np.int64)
    # labeled fixed
    # need to map labeled_labels back to positions
    labels[labeled_mask] = labeled_labels
    prev_unlab = None
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        dmat = np.linalg.norm(pool[unlabeled_idx][:, None, :] - centroids[None, :, :], axis=2)
        unlab_assign = np.argmin(dmat, axis=1)
        labels[unlabeled_idx] = unlab_assign
        if prev_unlab is not None and np.array_equal(unlab_assign, prev_unlab):
            break
        prev_unlab = unlab_assign.copy()
        for c in range(total_k):
            member = pool[labels == c]
            if len(member):
                centroids[c] = member.mean(axis=0)

    # inertia: sum squared distance to assigned centroid over whole pool
    dists = np.linalg.norm(pool - centroids[labels], axis=1)
    inertia = float(np.sum(dists**2))
    return labels, centroids, inertia, n_iter


def _best_by_silhouette(
    pool_unlabeled: np.ndarray,
    candidates: list[dict],
    total_k: int,
    **kwargs,
) -> tuple[int, list[float], list[bool], bool]:
    """Selezione best-silhouette con regola di scarto (spec §4A, pura).

    Nessuna label in input: la silhouette è calcolata sui soli unlabeled con
    le loro assegnazioni (sklearn, euclidea). Scarta i candidati con cluster
    singoletto o vuoto a livello pool, OPPURE con <total_k cluster non vuoti
    o singoletti sui 300 del test. Se tutti sono scartati, fallback
    pre-registrato = best-inertia (flag True).
    Qualsiasi kwarg (in particolare label-like) è rifiutato con TypeError.

    Ritorna (best_idx, silhouettes, discarded, fallback_alldiscarded);
    silhouettes vale NaN per i candidati scartati/non calcolabili.
    """
    from sklearn.metrics import silhouette_score

    if kwargs:
        raise TypeError(f"_best_by_silhouette rifiuta kwarg spuri {sorted(kwargs)} (anti-leakage: niente label)")
    if not candidates:
        raise ValueError("_best_by_silhouette richiede almeno un candidato")

    pool_unlabeled = np.asarray(pool_unlabeled, dtype=np.float64)
    n_init = len(candidates)
    silhouettes: list[float] = [float("nan")] * n_init
    discarded: list[bool] = [False] * n_init
    for i, cand in enumerate(candidates):
        pool_sizes = cand["cluster_sizes_pool"]
        test_sizes = cand["cluster_sizes_test"]
        if any(s <= 1 for s in pool_sizes):
            discarded[i] = True
            continue
        if sum(s > 0 for s in test_sizes) < total_k or any(s == 1 for s in test_sizes):
            discarded[i] = True
            continue
        lab_u = np.asarray(cand["labels"])[np.asarray(cand["unlabeled_idx"])]
        if len(set(lab_u.tolist())) < 2 or len(set(lab_u.tolist())) >= len(lab_u):
            discarded[i] = True
            continue
        try:
            silhouettes[i] = float(silhouette_score(pool_unlabeled, lab_u))
        except ValueError:
            discarded[i] = True
            silhouettes[i] = float("nan")
    fallback_alldiscarded = all(discarded)
    if fallback_alldiscarded:
        best_idx = int(np.argmin([c["inertia"] for c in candidates]))
    else:
        best_sil = float("-inf")
        best_idx = -1
        for i, cand in enumerate(candidates):
            if discarded[i]:
                continue
            if silhouettes[i] > best_sil:
                best_sil = silhouettes[i]
                best_idx = i
    assert best_idx >= 0  # invariante: candidates non vuoto (validato sopra) => best sempre assegnato
    return best_idx, silhouettes, discarded, fallback_alldiscarded


def density_novel_init(
    pool_norm: np.ndarray,
    unlabeled_idx: np.ndarray,
    known_centroids: np.ndarray,
    k: int = 10,
    pct: float = 90,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Seeding per densità (spec §4B, pura; solo 1 novel).

    Spazio L2-normalizzato, distanze euclidee. Densità(p) = distanza media ai
    k vicini più prossimi tra i soli unlabeled, ESCLUSO p stesso (k-NN
    standard; k=10 pre-registrato). Seme = argmin densità tra i punti oltre
    la soglia (p90 delle distanze al noto più vicino); centroide novel =
    media del suo k-NN (stesso vicinato, escluso sé). Insieme vuoto →
    fallback pre-registrato = media del k-NN del punto più denso IGNORANDO
    la soglia, con flag ``fallback_nosoglia: true``.

    Tie-break dell'argmin: a parità numerica (``np.isclose`` atol=1e-12) tra
    più punti a densità minima, ``rng.choice`` sceglie in modo deterministico
    a parità di seed; con ``rng=None`` vince il primo (``np.argmin``).
    Qualsiasi kwarg (in particolare label-like) è rifiutato con TypeError.

    Nessuna label in input (barriera anti-leakage strutturale).
    Ritorna (novel_centroid (D,), info).
    """
    from sklearn.neighbors import NearestNeighbors

    if kwargs:
        raise TypeError(f"density_novel_init rifiuta kwarg spuri {sorted(kwargs)} (anti-leakage: niente label)")
    pool_norm = np.asarray(pool_norm, dtype=np.float64)
    unlabeled_idx = np.asarray(unlabeled_idx)
    known_centroids = np.asarray(known_centroids, dtype=np.float64)
    Xu = pool_norm[unlabeled_idx]
    d_known = np.min(
        np.linalg.norm(Xu[:, None, :] - known_centroids[None, :, :], axis=2),
        axis=1,
    )
    threshold = float(np.percentile(d_known, pct))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(Xu)
    dists, neigh = nn.kneighbors(Xu)
    density = dists[:, 1:].mean(axis=1)
    beyond = np.where(d_known > threshold)[0]
    fallback_nosoglia = len(beyond) == 0
    pool_positions = beyond if not fallback_nosoglia else np.arange(len(Xu))
    pool_dens = density[pool_positions]
    best_d = float(np.min(pool_dens))
    tied = pool_positions[np.isclose(pool_dens, best_d, atol=1e-12)]
    if len(tied) > 1 and rng is not None:
        seed_pos = int(rng.choice(tied))
    else:
        seed_pos = int(tied[0])  # rng=None o unico minimo: first-wins deterministico
    centroid = Xu[neigh[seed_pos, 1 : k + 1]].mean(axis=0)
    info = {
        "fallback_nosoglia": bool(fallback_nosoglia),
        "threshold": threshold,
        "seed_unlabeled_pos": seed_pos,
        "k": int(k),
        "pct": float(pct),
    }
    return centroid, info


def osr_guided_novel_init(
    pool_norm: np.ndarray,
    unlabeled_idx: np.ndarray,
    known_centroids: np.ndarray,
    novelty_scores: np.ndarray,
    top_k: int = 50,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Seeding guidato OSR (spec §9 strategia D, pura; solo 1 novel).

    Classifica di novità KNN open-set (array di score, MAI label): i
    ``top_k`` punti unlabeled con score più alto definiscono il seme novel;
    centroide novel = media L2-normalizzata dei ``top_k`` (spazio
    L2-normalizzato). ``novelty_scores`` è allineato a ``unlabeled_idx``
    (un punteggio per unlabeled). Deterministico, un run.

    Nessuna label in input (barriera anti-leakage strutturale).
    Qualsiasi kwarg (in particolare label-like) è rifiutato con TypeError.

    Ritorna (novel_centroid (D,), info con top50_idx pool + top50_pos).
    """
    if kwargs:
        raise TypeError(f"osr_guided_novel_init rifiuta kwarg spuri {sorted(kwargs)} (anti-leakage: niente label)")
    pool_norm = np.asarray(pool_norm, dtype=np.float64)
    unlabeled_idx = np.asarray(unlabeled_idx)
    known_centroids = np.asarray(
        known_centroids, dtype=np.float64
    )  # interfaccia simmetrica a density_novel_init; non usato per la selezione
    novelty_scores = np.asarray(novelty_scores, dtype=np.float64).ravel()
    if novelty_scores.shape[0] != unlabeled_idx.shape[0]:
        raise ValueError(
            f"novelty_scores allineato a unlabeled_idx: {novelty_scores.shape[0]} != {unlabeled_idx.shape[0]}"
        )
    n_u = unlabeled_idx.shape[0]
    k = int(min(top_k, n_u))
    if k < 1:
        raise ValueError("top_k deve essere >=1 e servono unlabeled")
    order = np.argsort(-novelty_scores, kind="stable")
    top_pos = np.sort(order[:k])
    top_pool_idx = unlabeled_idx[top_pos]
    centroid = pool_norm[top_pool_idx].mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    info = {
        "top_k": k,
        "top50_pos": [int(p) for p in top_pos],
        "top50_idx": [int(i) for i in top_pool_idx],
    }
    return centroid, info


def ss_kmeans_multirun(
    pool_embeddings: np.ndarray,
    labeled_mask: np.ndarray,
    labeled_labels: np.ndarray | None = None,
    n_known: int | None = None,
    n_novel: int = 1,
    max_iter: int = 100,
    random_state: int = 57,
    n_init: int = 20,
    criterion: str = "inertia",
    return_candidates: bool = False,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Semi-supervised K-means vincolato, multi-restart (Vaze et al., CVPR 2022 — GCD).

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
        n_init: numero di init farthest-first (default 20).
        criterion: "inertia" (best-inertia, nessuno scarto) o "silhouette"
            (silhouette sklearn euclidea sui soli unlabeled, con regola di
            scarto pool+subset e fallback best-inertia — spec §4A).
        return_candidates: se True, info include "candidates" (un dict per
            init con labels, inertia, centroids, n_iter, cluster_sizes_pool,
            cluster_sizes_test, unlabeled_idx).

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
    if criterion not in ("inertia", "silhouette"):
        raise ValueError(f"criterion must be 'inertia' or 'silhouette', got {criterion!r}")

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

    candidates: list[dict] = []

    for init_idx in range(n_init):
        rng = np.random.default_rng(random_state + init_idx)

        # ---- Init centroidi novel (farthest-first con tie-break casuale) ----
        centroids_init = np.zeros((total_k, dim))
        centroids_init[:n_known] = known_centroids
        centroids_init[n_known:] = _farthest_first_novel(pool_embeddings[unlabeled_idx], known_centroids, n_novel, rng)

        # ---- Lloyd vincolato ----
        labels, centroids, inertia, n_iter = _constrained_lloyd(
            pool_embeddings, labeled_mask, labeled_labels, centroids_init, max_iter=max_iter
        )

        candidates.append(
            {
                "init_idx": int(init_idx),
                "labels": labels.copy(),
                "centroids": centroids.copy(),
                "inertia": float(inertia),
                "n_iter": int(n_iter),
                "cluster_sizes_pool": [int((labels == i).sum()) for i in range(total_k)],
                "cluster_sizes_test": [int((labels[unlabeled_idx] == i).sum()) for i in range(total_k)],
                "unlabeled_idx": unlabeled_idx.copy(),
            }
        )

    if criterion == "silhouette":
        best_init, silhouettes, discarded, fallback_alldiscarded = _best_by_silhouette(
            pool_embeddings[unlabeled_idx], candidates, total_k
        )
        best = candidates[best_init]
        best_labels = best["labels"]
        best_centroids = best["centroids"]
        best_inertia = best["inertia"]
        best_n_iter = best["n_iter"]
        best_sil = silhouettes[best_init]
    else:
        # scarto DISATTIVATO: path best-inertia verbatim (first-wins su <)
        best_labels = None
        best_centroids = None
        best_inertia = float("inf")
        best_n_iter = 0
        best_init = -1

        for cand in candidates:
            inertia = cand["inertia"]
            labels = cand["labels"]
            centroids = cand["centroids"]
            n_iter = cand["n_iter"]
            init_idx = cand["init_idx"]
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()
                best_n_iter = int(n_iter)
                best_init = int(init_idx)
        best_sil = float("nan")
        silhouettes = [float("nan")] * len(candidates)
        discarded = [False] * len(candidates)
        fallback_alldiscarded = False

    assert (
        best_labels is not None and best_centroids is not None
    )  # invariante: n_init>=1 validato => candidates non vuoto => best sempre assegnato
    info = {
        "n_iter": best_n_iter,
        "method": "ss_kmeans_constrained_vaze2022",
        "n_known": n_known,
        "n_novel": n_novel,
        "cluster_sizes": [int((best_labels == i).sum()) for i in range(total_k)],
        "inertia": float(best_inertia),
        "n_init": int(n_init),
        "best_init": int(best_init),
        "criterion": criterion,
        "silhouette": float(best_sil),
        "silhouettes": [float(s) for s in silhouettes],
        "discarded": [bool(x) for x in discarded],
        "n_discarded": int(sum(1 for x in discarded if x)),
        "fallback_alldiscarded": bool(fallback_alldiscarded),
    }
    if return_candidates:
        info["candidates"] = candidates
    return best_labels, info


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

    Wrapper di :func:`ss_kmeans_multirun` con ``criterion="inertia"``
    (best-inertia, nessuno scarto): risultati equivalenti (verificato
    pre/post su dati reali).

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
        n_init: numero di init farthest-first (default 20).

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
    return ss_kmeans_multirun(
        pool_embeddings,
        labeled_mask,
        labeled_labels,
        n_known=n_known,
        n_novel=n_novel,
        max_iter=max_iter,
        random_state=random_state,
        n_init=n_init,
        criterion="inertia",
        return_candidates=False,
    )


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
