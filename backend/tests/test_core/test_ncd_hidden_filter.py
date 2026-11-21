"""Il train NCD deve escludere le classi nascoste e rinumerare le label.

Senza rinumerazione la maschera `label < num_classes` della GCD scambia noti e
novel: e' successo davvero, con ACC new 0.996 su una specie che il modello non
aveva mai visto.
"""

from types import SimpleNamespace

import numpy as np

from app.core.ncd.pipeline import drop_hidden_classes


def _bundle(names, labels, dim=3):
    n = len(labels)
    return SimpleNamespace(
        train_embeddings=np.arange(n * dim, dtype=float).reshape(n, dim),
        train_labels=np.array(labels, dtype=np.int64),
        train_class_names=list(names),
    )


def test_drops_the_hidden_class_and_renumbers():
    bundle = _bundle(["A", "N", "B"], [0, 1, 2, 0, 1, 2])
    out = drop_hidden_classes(bundle, ["N"])
    assert out.train_embeddings.shape[0] == 4
    assert out.train_labels.tolist() == [0, 1, 0, 1]  # contigue, senza buco
    assert out.train_class_names == ["A", "B"]


def test_real_six_classes_drop_garganica():
    names = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]
    labels = [0, 1, 2, 3, 4, 5] * 2
    out = drop_hidden_classes(_bundle(names, labels), ["O. garganica"])
    assert "O. garganica" not in out.train_class_names
    assert set(np.unique(out.train_labels)) == {0, 1, 2, 3, 4}
    assert max(out.train_labels) < len(out.train_class_names)


def test_no_exclusions_is_a_noop():
    bundle = _bundle(["A", "B"], [0, 1, 0, 1])
    out = drop_hidden_classes(bundle, [])
    assert out.train_labels.tolist() == [0, 1, 0, 1]
    assert out.train_class_names == ["A", "B"]


def test_two_hidden_classes():
    names = ["A", "N1", "B", "N2"]
    labels = [0, 1, 2, 3] * 2
    out = drop_hidden_classes(_bundle(names, labels), ["N1", "N2"])
    assert out.train_class_names == ["A", "B"]
    assert out.train_labels.tolist() == [0, 1, 0, 1]


def test_upload_ncd_usa_i_nomi_canonici():
    """Upload e API devono usare lo stesso nome: il sync appiattisce una volta.

    La mappa exp4→exp9_ncd ha spedito i risultati NCD di II-A nel posto
    sbagliato (e sovrascritto i dati della tesi).
    """
    from app.hf.ncd import MODELS

    for local_prefix, remote_prefix in MODELS:
        assert local_prefix == remote_prefix, f"{local_prefix} → {remote_prefix}"
