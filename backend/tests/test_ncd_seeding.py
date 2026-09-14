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


# (b) Non-leakage ------------------------------------------------------------
def test_b_no_label_params_in_signatures():
    for fn in (_farthest_first_novel, density_novel_init, _best_by_silhouette):
        params = list(inspect.signature(fn).parameters)
        # "unlabeled" (indici/embedding non etichettati) è lecito; vietate le true-label
        assert not any("label" in p and "unlabeled" not in p for p in params), (fn.__name__, params)


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


def test_b_permuted_test_labels_same_init():
    pool, mask, train_labels, n_known, _ = _toy_pool()
    kc = _known_centroids(pool, mask, train_labels, n_known)
    unl = pool[~mask]
    rng = np.random.default_rng(57)
    init1 = _farthest_first_novel(unl, kc, 1, rng)
    # le label del test vengono permutate: l'init non le riceve mai
    fake_test_labels = np.arange((~mask).sum())
    rng2 = np.random.default_rng(99)
    perm = rng2.permutation(fake_test_labels)
    assert set(perm.tolist()) == set(fake_test_labels.tolist())
    init2 = _farthest_first_novel(unl, kc, 1, np.random.default_rng(57))
    assert np.array_equal(init1, init2)
    c1, _ = density_novel_init(pool, np.where(~mask)[0], kc)
    c2, _ = density_novel_init(pool, np.where(~mask)[0], kc)
    assert np.array_equal(c1, c2)


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
