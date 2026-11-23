"""Test della mappatura delle label nell'ablazione SpHOR.

Il punto delicato: ImageFolder numera le classi in ordine alfabetico sulle
cartelle presenti, il modello le numera secondo TRAIN_CLASSES. Se la mappatura
sbaglia, l'AUROC e' rumore — e non te ne accorgi.
"""

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ablation_sphor.py"


def _load():
    spec = importlib.util.spec_from_file_location("ablation_sphor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ab = _load()


class TestMapLabels:
    def test_drops_classes_the_model_does_not_have(self):
        labels = np.array([0, 1, 2])
        mapped, keep = ab.map_labels(labels, ["a", "b", "novel"], ["a", "b"])
        assert keep.tolist() == [True, True, False]
        assert mapped.tolist() == [0, 1]

    def test_remaps_when_the_alphabetical_order_differs(self):
        labels = np.array([0, 1])
        mapped, keep = ab.map_labels(labels, ["b", "a"], ["a", "b"])
        assert keep.tolist() == [True, True]
        assert mapped.tolist() == [1, 0]

    def test_all_unknown_gives_empty(self):
        mapped, keep = ab.map_labels(np.array([0, 1]), ["novel", "other"], ["a", "b"])
        assert keep.tolist() == [False, False]
        assert mapped.size == 0

    def test_real_case_studio_iia(self):
        dataset = [
            "O. exaltata",
            "O. garganica",
            "O. incubacea",
            "O. majellensis",
            "O. sphegodes",
            "O. sphegodes_Palena",
        ]
        model = [c for c in dataset if c != "O. garganica"]
        mapped, keep = ab.map_labels(np.arange(6), dataset, model)
        assert keep.tolist() == [True, False, True, True, True, True]
        # garganica resta fuori, le altre mantengono l'ordine del modello
        assert mapped.tolist() == [0, 1, 2, 3, 4]
