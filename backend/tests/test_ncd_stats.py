"""Tests for app.core.ncd.stats (synthetic embeddings, no GPU)."""

import numpy as np


def _make_emb_exp(base, name, n_classes=2, dim=8, per_class=20):
    import os

    emb_dir = os.path.join(base, name, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    rng = np.random.default_rng(57)
    embs, labels = [], []
    for c in range(n_classes):
        embs.append(rng.normal(c * 5.0, 1.0, (per_class, dim)))
        labels += [c] * per_class
    np.save(os.path.join(emb_dir, "embeddings.npy"), np.vstack(embs).astype(np.float32))
    np.save(os.path.join(emb_dir, "labels.npy"), np.array(labels))
    return os.path.join(base, name)


def test_process_writes_npz(tmp_path):
    import os

    from app.core.ncd.stats import process

    _make_emb_exp(str(tmp_path), "exp4_sphor_ncd_resnet18")
    out = os.path.join(str(tmp_path), "stats")
    r = process("exp4_sphor_ncd_resnet18", exp_base=str(tmp_path), out_dir=out, n_classes=2)
    assert r is not None
    assert os.path.exists(os.path.join(out, "exp4_sphor_ncd_resnet18.npz"))
    z = np.load(os.path.join(out, "exp4_sphor_ncd_resnet18.npz"))
    for k in ["class_means", "cov_inv", "train_embeddings", "threshold_ensemble", "maha_mean", "knn_mean"]:
        assert k in z.files


def test_process_missing_embeddings_returns_none(tmp_path):
    import os

    from app.core.ncd.stats import process

    os.makedirs(os.path.join(str(tmp_path), "exp9_x"))
    assert process("exp9_x", exp_base=str(tmp_path), out_dir=str(tmp_path)) is None
