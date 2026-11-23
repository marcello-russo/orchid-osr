# tests/test_api/test_serializers.py
"""Unit test dei serializer: nessun client HTTP, solo JSON su disco → payload.

Le route sono I/O e validazione; le regole di dominio (quale F1 è ufficiale,
come si normalizzano i fold, le ricette) vivono in `app/api/serializers/` e si
possono verificare direttamente.
"""

import json

import pytest

from app.api.serializers.experiments import (
    RECIPES,
    load_json,
    override_test_metrics,
    perfold_stats,
    read_ncd_metrics,
)
from app.api.serializers.folds import normalize_fold


class TestNormalizeFold:
    def test_single_phase_truncates_history_after_patience(self):
        fold = {
            "fold": 0,
            "best_epoch": 3,
            "best_f1": 0.9,
            "best_accuracy": 0.91,
            "history": {
                "train_loss": [1.0] * 40,
                "val_loss": [0.5] * 40,
                "val_f1": [0.8] * 40,
                "val_accuracy": [0.85] * 40,
            },
        }
        out = normalize_fold(fold)
        # best_epoch 3 (0-based) + patience 15 → 18 epoche servite (indici 0..17)
        assert len(out["history"]["train_loss"]) == 18
        assert out["best_epoch"] == 17
        assert out["fold_phases"] is None
        # l'input non viene mutato
        assert len(fold["history"]["train_loss"]) == 40

    def test_two_phase_exposes_supcon_as_first_phase(self):
        fold = {
            "fold": 1,
            "best_epoch": 2,
            "history": {
                "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
                "val_loss": [0.9, 0.7, 0.5, 0.4, 0.3],
                "val_f1": [0.5, 0.6, 0.7, 0.7, 0.7],
                "val_accuracy": [0.6, 0.7, 0.75, 0.75, 0.75],
            },
            "supcon_phase": {
                "supcon_history": {
                    "train_loss": [2.0, 1.5],
                    "val_knn_acc": [0.4, 0.6],
                }
            },
        }
        out = normalize_fold(fold)
        assert out["fold_phases"][0]["type"] == "contrastive"
        assert out["fold_phases"][0]["epochs"] == 2
        assert out["fold_phases"][0]["metrics"]["val_knn_acc"] == [0.4, 0.6]
        # Phase 2 troncata a best_epoch + patience
        assert len(out["history"]["train_loss"]) == 5
        assert "supcon_phase" not in out

    def test_already_normalized_fold_is_returned_as_is(self):
        fold = {"fold": 0, "fold_phases": [], "history": {"train_loss": [1.0]}}
        assert normalize_fold(fold) is fold


class TestRecipes:
    def test_every_experiment_has_a_recipe(self):
        assert set(RECIPES) == set(range(1, 11))
        for num, recipe in RECIPES.items():
            assert recipe["phases"], num
            assert recipe["recipe"], num


class TestPerfoldStats:
    def test_reads_mean_and_std(self, tmp_path):
        (tmp_path / "perfold_test.json").write_text(json.dumps({"f1_mean": 0.87, "f1_std": 0.02}))
        assert perfold_stats(tmp_path) == (0.87, 0.02)

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert perfold_stats(tmp_path) == (None, None)

    def test_malformed_file_is_not_an_error(self, tmp_path):
        (tmp_path / "perfold_test.json").write_text(json.dumps({"f1_mean": "tanto"}))
        assert perfold_stats(tmp_path) == (None, None)


class TestOverrideTestMetrics:
    def test_confusion_matrix_becomes_official_test_metrics(self, tmp_path):
        (tmp_path / "perfold_test.json").write_text(
            json.dumps(
                {
                    "best_confusion_matrix": [[8, 2], [1, 9]],
                    "best_confusion_matrix_labels": ["A", "B"],
                }
            )
        )
        results = {"test_metrics": {"accuracy": 0.0, "f1_macro": 0.0}}
        override_test_metrics(tmp_path, results)
        metrics = results["test_metrics"]
        assert metrics["accuracy"] == pytest.approx(17 / 20)
        assert metrics["confusion_matrix"] == [[8, 2], [1, 9]]
        assert metrics["confusion_matrix_labels"] == ["A", "B"]
        # precision/recall/F1 per classe ricalcolati dalla matrice
        assert metrics["per_class_metrics"]["A"]["precision"] == pytest.approx(8 / 9)
        assert metrics["per_class_metrics"]["A"]["recall"] == pytest.approx(8 / 10)

    def test_without_perfold_file_nothing_changes(self, tmp_path):
        results = {"test_metrics": {"accuracy": 0.5}}
        override_test_metrics(tmp_path, results)
        assert results["test_metrics"] == {"accuracy": 0.5}

    def test_none_results_is_a_noop(self, tmp_path):
        override_test_metrics(tmp_path, None)


class TestLoadJson:
    def test_non_object_json_is_rejected(self, tmp_path):
        path = tmp_path / "lista.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="oggetto JSON"):
            load_json(path)


class TestReadNcdMetrics:
    def test_ensemble_wins_over_better_single_detector(self, tmp_path):
        (tmp_path / "ncd_results.json").write_text(
            json.dumps(
                {
                    "detectors": {
                        "Maha": {"auroc": 0.90},
                        "Ensemble": {"auroc": 0.85},
                    },
                    "clustering": {"KMeans": {"nmi": 0.5}},
                }
            )
        )
        metrics = read_ncd_metrics(tmp_path)
        # Regola della tesi: Ensemble, anche se un singolo rivelatore segna di più
        assert metrics["ncd_detector"] == "Ensemble"
        assert metrics["ncd_auroc"] == 0.85
        assert metrics["ncd_clustering_nmi"] == 0.5

    def test_best_detector_is_declared_when_ensemble_is_missing(self, tmp_path):
        (tmp_path / "ncd_results.json").write_text(
            json.dumps({"detectors": {"Maha": {"auroc": 0.90}, "Energy": {"auroc": 0.70}}})
        )
        metrics = read_ncd_metrics(tmp_path)
        assert metrics["ncd_detector"] == "Maha"

    def test_missing_file_gives_empty_metrics(self, tmp_path):
        assert read_ncd_metrics(tmp_path) == {}
