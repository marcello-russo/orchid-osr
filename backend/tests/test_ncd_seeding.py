"""Test FASE 1 — GCD novel-centroid seeding (piano §4, test a-e)."""

import inspect

import numpy as np
import pytest

from app.core.ncd.clustering import (
    _best_by_silhouette,
    _farthest_first_novel,
    build_labeled_mask,
    density_novel_init,
    ss_kmeans_clustering,
    ss_kmeans_multirun,
)


def _toy_pool(seed=0):
    rng = np.random.default_rng(seed)
    n_known, n_novel, dim = 3, 1, 8
    centers = rng.normal(size=(n_known + n_novel, dim)) * 5.0
    train = np.vstack([c + rng.normal(scale=0.5, size=(20, dim)) for c in centers[:n_known]])
    train_labels = np.repeat(np.arange(n_known), 20)
    test = np.vstack(
        [c + rng.normal(scale=0.5, size=(8, dim)) for c in centers[:n_known]]
        + [centers[n_known] + rng.normal(scale=0.5, size=(6, dim))]
    )
    pool = np.vstack([train, test])
    mask = build_labeled_mask(len(train), len(test))
    return pool, mask, train_labels, n_known, n_novel


def _known_centroids(pool, mask, train_labels, n_known):
    kc = np.zeros((n_known, pool.shape[1]))
    for c in range(n_known):
        kc[c] = pool[mask][train_labels == c].mean(axis=0)
    return kc


# (a) Regressione esatta wrapper vs multirun-inertia (stesso RNG path) ------
def test_a_regression_exact():
    pool, mask, train_labels, n_known, n_novel = _toy_pool()
    lab_w, info_w = ss_kmeans_clustering(
        pool, mask, train_labels, n_known=n_known, n_novel=n_novel, random_state=57
    )
    lab_m, info_m = ss_kmeans_multirun(
        pool, mask, train_labels, n_known=n_known, n_novel=n_novel,
        random_state=57, criterion="inertia",
    )
    assert np.array_equal(lab_w, lab_m)
    assert info_w["inertia"] == info_m["inertia"]
    for key in ("n_iter", "best_init", "cluster_sizes", "n_init", "method"):
        assert info_w[key] == info_m[key]
    # determinismo a parità di seed
    lab2, info2 = ss_kmeans_clustering(
        pool, mask, train_labels, n_known=n_known, n_novel=n_novel, random_state=57
    )
    assert np.array_equal(lab_w, lab2)
    assert info_w["inertia"] == info2["inertia"]


# (a2) Regressione golden pre/post refactor -----------------------------------
# Golden generati col codice pre-refactor (git show 60425c0:.../clustering.py
# caricato come modulo temp) su array sintetico seedato:
#   rng = default_rng(0); pool = rng.normal(size=(24, 4)); mask[:16]=True;
#   labels = repeat([0,1], 8); ss_kmeans_clustering(..., n_known=2, n_novel=1,
#   random_state=57)
_GOLDEN_LABELS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 0, 2, 1, 0, 1, 1, 0]
_GOLDEN_INERTIA = 67.63177432041103
_GOLDEN_BEST_INIT = 0
_GOLDEN_N_ITER = 2
_GOLDEN_SIZES = [11, 12, 1]


def _golden_pool():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(24, 4))
    mask = np.zeros(24, dtype=bool)
    mask[:16] = True
    labels = np.repeat([0, 1], 8)
    return pool, mask, labels


def test_a2_golden_pre_post_refactor():
    pool, mask, train_labels = _golden_pool()
    for fn in (ss_kmeans_clustering, ss_kmeans_multirun):
        lab, info = fn(pool, mask, train_labels, n_known=2, n_novel=1, random_state=57)
        assert lab.tolist() == _GOLDEN_LABELS
        assert info["inertia"] == _GOLDEN_INERTIA
        assert info["best_init"] == _GOLDEN_BEST_INIT
        assert info["n_iter"] == _GOLDEN_N_ITER
        assert info["cluster_sizes"] == _GOLDEN_SIZES


# (b) Non-leakage ------------------------------------------------------------
def test_b_no_label_params_in_signatures():
    for fn in (_farthest_first_novel, density_novel_init, _best_by_silhouette):
        params = list(inspect.signature(fn).parameters)
        # "unlabeled" (indici/embedding non etichettati) è lecito; vietate le true-label
        assert not any("label" in p and "unlabeled" not in p for p in params), (fn.__name__, params)


def test_b_label_kwargs_rejected():
    """Rifiuto esplicito di kwarg label-like (non-leakage non vacuo)."""
    pool, mask, train_labels, n_known, _ = _toy_pool()
    kc = _known_centroids(pool, mask, train_labels, n_known)
    unl = pool[~mask]
    rng = np.random.default_rng(57)
    with pytest.raises(TypeError):
        _farthest_first_novel(unl, kc, 1, rng, test_labels=np.zeros(len(unl)))
    with pytest.raises(TypeError):
        density_novel_init(pool, np.where(~mask)[0], kc, y=np.zeros(len(pool)))
    lab, info = ss_kmeans_multirun(
        pool, mask, train_labels, n_known=n_known, n_novel=1,
        random_state=57, criterion="inertia", return_candidates=True,
    )
    with pytest.raises(TypeError):
        _best_by_silhouette(unl, info["candidates"], n_known + 1, true_labels=np.zeros(len(unl)))


def test_b_mask_builder_pure():
    m1 = build_labeled_mask(60, 30)
    assert m1.dtype == bool and m1.sum() == 60
    assert m1[:60].all() and not m1[60:].any()
    # specie rimescolate -> stessa maschera (dipende solo dalle cardinalità)
    rng = np.random.default_rng(1)
    species = np.array(["a"] * 60 + ["b"] * 30)
    assert np.array_equal(build_labeled_mask(60, 30), m1)
    assert np.array_equal(build_labeled_mask(60, 30), build_labeled_mask(60, 30))
    _ = rng.permutation(species)  # le label non entrano nel builder


def test_b_determinism_same_seed():
    """Determinismo a parità di seed (ex test tautologico sulle label permutate:
    le label permutate non entravano mai nell'init, quindi il test provava
    solo determinismo — ora dichiarato esplicitamente)."""
    pool, mask, train_labels, n_known, _ = _toy_pool()
    kc = _known_centroids(pool, mask, train_labels, n_known)
    unl = pool[~mask]
    init1 = _farthest_first_novel(unl, kc, 1, np.random.default_rng(57))
    init2 = _farthest_first_novel(unl, kc, 1, np.random.default_rng(57))
    assert np.array_equal(init1, init2)
    c1, _ = density_novel_init(pool, np.where(~mask)[0], kc)
    c2, _ = density_novel_init(pool, np.where(~mask)[0], kc)
    assert np.array_equal(c1, c2)
    # rng diverso -> tie-break potenzialmente diverso ma sempre deterministico
    c3, _ = density_novel_init(pool, np.where(~mask)[0], kc, rng=np.random.default_rng(57))
    c4, _ = density_novel_init(pool, np.where(~mask)[0], kc, rng=np.random.default_rng(57))
    assert np.array_equal(c3, c4)


# (c) Docstring n_init --------------------------------------------------------
def test_c_docstring_n_init():
    for fn in (ss_kmeans_clustering, ss_kmeans_multirun):
        assert inspect.signature(fn).parameters["n_init"].default == 20
        assert "(default 20)" in (fn.__doc__ or "")


# (d) Candidati esposti --------------------------------------------------------
def test_d_candidates_first_wins():
    pool, mask, train_labels, n_known, n_novel = _toy_pool()
    n_init = 20
    labels, info = ss_kmeans_multirun(
        pool, mask, train_labels, n_known=n_known, n_novel=n_novel,
        random_state=57, criterion="inertia", return_candidates=True,
    )
    cands = info["candidates"]
    assert len(cands) == n_init
    inertias = [c["inertia"] for c in cands]
    assert all(np.isfinite(inertias))
    # best == argmin first-wins
    assert info["best_init"] == int(np.argmin(inertias))
    assert np.array_equal(labels, cands[info["best_init"]]["labels"])
    assert info["inertia"] == min(inertias)
    total_k = n_known + n_novel
    for c in cands:
        lab = c["labels"]
        assert [int((lab == i).sum()) for i in range(total_k)] == c["cluster_sizes_pool"]
        assert sum(c["cluster_sizes_pool"]) == len(pool)
        assert sum(c["cluster_sizes_test"]) == int((~mask).sum())
    assert info["cluster_sizes"] == cands[info["best_init"]]["cluster_sizes_pool"]
    assert not info["fallback_alldiscarded"] and info["n_discarded"] == 0


# (e) Fallback B ----------------------------------------------------------------
def test_e_density_fallback_nosoglia():
    pool, mask, train_labels, n_known, _ = _toy_pool()
    kc = _known_centroids(pool, mask, train_labels, n_known)
    unl_idx = np.where(~mask)[0]
    centroid, info = density_novel_init(pool, unl_idx, kc, pct=100)
    assert info["fallback_nosoglia"] is True
    assert np.all(np.isfinite(centroid)) and centroid.shape == (pool.shape[1],)
    centroid_ok, info_ok = density_novel_init(pool, unl_idx, kc, pct=90)
    assert info_ok["fallback_nosoglia"] is False
    assert np.all(np.isfinite(centroid_ok))


def test_criterion_invalid():
    pool, mask, train_labels, n_known, n_novel = _toy_pool()
    with pytest.raises(ValueError, match="criterion"):
        ss_kmeans_multirun(
            pool, mask, train_labels, n_known=n_known, n_novel=n_novel,
            criterion="bogus",
        )


# (f) Fallback A: tutti scartati -> flag + best-inertia -------------------------
def test_f_fallback_alldiscarded():
    """Tutti i candidati scartati (singoletti a livello pool) -> fallback
    best-inertia con flag True (spec §4A)."""
    rng = np.random.default_rng(3)
    unl = rng.normal(size=(8, 4))
    n_init = 4
    cands = []
    for i in range(n_init):
        lab = np.zeros(8, dtype=np.int64)
        lab[i] = 1  # cluster 1 singoletto -> scarto pool
        cands.append({
            "labels": lab,
            "inertia": 100.0 + i * 10.0,
            "cluster_sizes_pool": [7, 1],
            "cluster_sizes_test": [7, 1],
            "unlabeled_idx": np.arange(8),
        })
    best_idx, sils, discarded, fallback = _best_by_silhouette(unl, cands, 2)
    assert fallback is True
    assert all(discarded)
    assert all(np.isnan(s) for s in sils)
    # fallback = best-inertia = init 0
    assert best_idx == 0


# (g) SeedSequence C: 10 distinti, range sklearn --------------------------------
def test_g_seedsequence_c_seeds():
    """Schema seed controllo C (spec §4C): 10 init distinti < 2**31 per ogni g."""
    for g in range(57, 67):
        seeds = [int(np.random.SeedSequence(g).spawn(10)[j].generate_state(1)[0]) % 2**31
                 for j in range(10)]
        assert len(set(seeds)) == 10
        assert all(0 <= s < 2**31 for s in seeds)


# (h) pool_hungarian_test_acc: slice posizionale, mapping pool ------------------
def test_h_pool_hungarian_test_acc():
    """Mapping Hungarian fittato sul pool, ACC valutate sulla slice test
    posizionale dopo n_train (verbatim gcd_analysis.py:180-189)."""
    from app.core.ncd.evaluation import pool_hungarian_test_acc

    n_train = 20
    pool_true = np.array([0] * 10 + [1] * 10 + [0] * 5 + [1] * 5)
    # pred = true con cluster swappati (0<->1): mapping pool deve recuperarli
    pred = np.array([1] * 10 + [0] * 10 + [1] * 5 + [0] * 5)
    test_known_mask = np.array([True] * 5 + [False] * 5)
    out = pool_hungarian_test_acc(pool_true, pred, n_train, test_known_mask)
    assert out["acc_all_test"] == 1.0
    assert out["acc_old_test"] == 1.0
    assert out["acc_new_test"] == 1.0
    assert out["n_old_test"] == 5 and out["n_new_test"] == 5
    assert out["acc_all_pool"] == 1.0


# (i) Tie inerzia: first-wins ----------------------------------------------------
def test_i_inertia_tie_first_wins():
    """A parità di inerzia vince il primo init: il path best-inertia usa
    ``<`` stretto (verbatim) e ``np.argmin`` è first-wins."""
    import inspect

    from app.core.ncd import clustering as cl

    assert "if inertia < best_inertia:" in inspect.getsource(cl.ss_kmeans_multirun)
    assert int(np.argmin(np.array([5.0, 5.0, 7.0]))) == 0
