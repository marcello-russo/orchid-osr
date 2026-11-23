"""Test per l'endpoint GET /api/experiments/{id}."""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


@pytest.fixture
def _fake_experiments_dir(tmp_path, monkeypatch):
    """Sostituisce experiments dir con una temporanea."""
    from app.api.routes import experiments as exp_route

    monkeypatch.setattr(exp_route, "_get_experiments_dir", lambda: tmp_path)
    return tmp_path


def _make_exp(base_dir, name, config, results, folds=5):
    """Crea un experimento finto con config.json, results.json e fold files."""
    d = base_dir / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    (d / "results.json").write_text(json.dumps(results))
    for i in range(folds):
        (d / f"fold_{i}.json").write_text(
            json.dumps(
                {
                    "best_f1": 0.8 + i * 0.02,
                    "fold": i,
                    "best_accuracy": 0.78 + i * 0.02,
                    "best_epoch": 10,
                    "history": {
                        "train_loss": [0.5, 0.3, 0.2],
                        "val_loss": [0.6, 0.4, 0.3],
                        "val_accuracy": [0.7, 0.8, 0.82],
                        "val_f1": [0.68, 0.78, 0.8],
                    },
                    "train_size": 80,
                    "val_size": 20,
                }
            )
        )


class TestExperimentDetail:
    """Test per GET /api/experiments/{id}."""

    def test_detail_existing(self, _fake_experiments_dir):
        """GET con esperimento esistente restituisce 200 con config e folds."""
        _make_exp(
            _fake_experiments_dir,
            "exp6_clean_resnet18",
            {"model_variant": "resnet18", "num_classes": 6, "timestamp": "2026-01-01T00:00:00"},
            {
                "mean_f1": 0.85,
                "std_f1": 0.02,
                "mean_accuracy": 0.83,
                "test_metrics": {"accuracy": 0.79, "f1_macro": 0.82},
                "best_fold": 0,
            },
            folds=5,
        )

        resp = client.get("/api/experiments/exp6_clean_resnet18")
        assert resp.status_code == 200
        data = resp.json()

        assert data["id"] == "exp6_clean_resnet18"
        assert data["config"]["model_variant"] == "resnet18"
        assert len(data["folds"]) == 5
        assert data["results"]["mean_f1"] == 0.85
        assert data["summary"]["num_folds"] == 5

    def test_detail_not_found(self, _fake_experiments_dir):
        """GET con esperimento inesistente restituisce 404."""
        resp = client.get("/api/experiments/nonexistent")
        assert resp.status_code == 404

    def test_detail_without_folds(self, _fake_experiments_dir):
        """GET con esperimento senza fold_*.json restituisce 200 con folds vuoto."""
        d = _fake_experiments_dir / "exp6_no_folds"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_variant": "resnet18"}))
        (d / "results.json").write_text(
            json.dumps(
                {
                    "mean_f1": 0.8,
                    "std_f1": 0.02,
                    "mean_accuracy": 0.78,
                    "test_metrics": {"accuracy": 0.77, "f1_macro": 0.79},
                    "best_fold": 0,
                }
            )
        )

        resp = client.get("/api/experiments/exp6_no_folds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "exp6_no_folds"
        assert data["folds"] == []
        assert data["summary"]["num_folds"] == 0
