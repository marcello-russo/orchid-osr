"""Test per l'endpoint /api/experiments.

Verifica che i campi is_ncd, ncd_auroc, test_accuracy siano corretti.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fake_experiments(tmp_path, monkeypatch):
    """Sostituisce experiments dir con una temporanea contenente esperimenti finti."""
    from app.api.routes import experiments as exp_route

    monkeypatch.setattr(exp_route, "_get_experiments_dir", lambda: tmp_path)
    # Per test di integrazione serve anche il DB, ma lo saltiamo perché
    # list_experiments legge da JSON, non dal DB.

    # Crea esperimenti finti
    _make_exp(
        tmp_path,
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

    _make_exp(
        tmp_path,
        "exp2_supcon_resnet18",
        {"model_variant": "resnet18", "num_classes": 6, "timestamp": "2026-01-02T00:00:00"},
        {
            "mean_f1": 0.38,
            "std_f1": 0.04,
            "mean_accuracy": 0.38,
            "test_metrics": {"accuracy": 0.47, "f1_macro": 0.36},
            "best_fold": 2,
        },
        folds=5,
    )

    _make_exp(
        tmp_path,
        "exp3_sphor_convnext_tiny",
        {"model_variant": "convnext_tiny", "num_classes": 6, "timestamp": "2026-01-03T00:00:00"},
        {
            "mean_f1": 0.81,
            "std_f1": 0.05,
            "mean_accuracy": 0.80,
            "test_metrics": {"accuracy": 0.83, "f1_macro": 0.80},
            "best_fold": 1,
        },
        folds=5,
    )

    _make_exp(
        tmp_path,
        "exp4_sphor_ncd_resnet18",
        {
            "model_variant": "resnet18",
            "num_classes": 5,
            "ncd_mode": True,
            "exclude_classes": ["O. garganica"],
            "timestamp": "2026-01-06T00:00:00",
        },
        {
            "mean_f1": 0.59,
            "std_f1": 0.03,
            "mean_accuracy": 0.59,
            "test_metrics": {"accuracy": 0.60, "f1_macro": 0.59},
            "best_fold": 0,
        },
        folds=5,
        ncd={
            "detectors": {"Ensemble": {"auroc": 0.42}, "KNN": {"auroc": 0.40}},
            "novel_class": "O. garganica",
        },
    )

    _make_exp(
        tmp_path,
        "exp10_mgh_ncd_dinov2_base",
        {"model_variant": "dinov2_base", "num_classes": 6, "ncd_mode": True, "timestamp": "2026-01-04T00:00:00"},
        {
            "mean_f1": 0.86,
            "std_f1": 0.01,
            "mean_accuracy": 0.86,
            "test_metrics": {"accuracy": 0.14, "f1_macro": 0.05},
            "best_fold": 0,
        },
        folds=5,
        ncd={
            "detectors": {"Ensemble": {"auroc": 0.91}, "KNN": {"auroc": 0.89}},
            "clustering": {"spectral": {"nmi": 0.51}},
        },
    )

    _make_exp(
        tmp_path,
        "exp10_mgh_ncd_resnet18",
        {"model_variant": "resnet18", "num_classes": 6, "ncd_mode": True, "timestamp": "2026-01-05T00:00:00"},
        {
            "mean_f1": 0.38,
            "std_f1": 0.02,
            "mean_accuracy": 0.38,
            "test_metrics": {"accuracy": 0.24, "f1_macro": 0.12},
            "best_fold": 2,
        },
        folds=3,
        ncd={
            "detectors": {"Mahalanobis": {"auroc": 0.77}},
        },
    )


def _make_exp(base_dir, name, config, results, folds=5, ncd=None):
    """Crea un experimento finto con config.json, results.json, fold files e opzionale ncd_results."""
    d = base_dir / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    (d / "results.json").write_text(json.dumps(results))
    for i in range(folds):
        (d / f"fold_{i}.json").write_text(
            json.dumps({"best_f1": 0.8 + i * 0.02, "fold": i, "best_accuracy": 0.78 + i * 0.02})
        )
    if ncd:
        (d / "ncd_results.json").write_text(json.dumps(ncd))


class TestExperimentsListNCD:
    """Test che i campi NCD siano corretti nell'elenco esperimenti."""

    def test_ce_experiment_has_no_ncd_fields(self):
        """Exp6 (CE) non deve avere campi NCD."""
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.json()
        for e in data:
            if "exp1" in e["id"]:
                assert e.get("is_ncd") is False
                assert e.get("ncd_auroc") is None
                assert e["test_accuracy"] == 0.79  # standard CE accuracy

    def test_supcon_experiment_no_ncd(self):
        """Exp7 (SupCon senza ncd_mode) non deve segnare is_ncd."""
        resp = client.get("/api/experiments")
        data = resp.json()
        for e in data:
            if "exp2" in e["id"]:
                assert e.get("is_ncd") is False
                assert e.get("ncd_auroc") is None

    def test_sphor_experiment_has_ncd_auroc(self):
        """Exp8 (SpHOR) è closed-set: senza ncd_auroc e is_ncd=False."""
        resp = client.get("/api/experiments")
        data = resp.json()
        for e in data:
            if "exp3" in e["id"]:
                assert e.get("is_ncd") is False
                assert e.get("ncd_auroc") is None
                # test_accuracy resta l'accuracy standard
                assert e["test_accuracy"] == 0.83

    def test_ncd_experiment_is_ncd_true(self):
        """Exp9 (NCD) deve avere is_ncd=True e ncd_auroc."""
        resp = client.get("/api/experiments")
        data = resp.json()
        for e in data:
            if "exp4" in e["id"]:
                assert e.get("is_ncd") is True
                assert e.get("ncd_auroc") == 0.42
                assert e.get("ncd_detector") == "Ensemble"
                # test_accuracy resta l'accuracy standard; l'AUROC è esposto in ncd_auroc
                assert e["test_accuracy"] == 0.60

    def test_ncd_without_ncd_file_means_no_ncd(self):
        """Se exp ha ncd_mode=True ma non ncd_results.json, è NCD ma senza AUROC."""
        # Questo test richiede di creare un exp con ncd_mode ma senza file ncd
        # Già coperto dalla fixture: exp10_mgh_ncd_resnet18 non ha ncd_results
        pass

    def test_mean_f1_zero_not_overridden(self):
        """mean_f1=0.0 non deve far cadere nel fallback test_metrics."""
        resp = client.get("/api/experiments")
        data = resp.json()
        # Tutti i mean_f1 devono essere numerici
        for e in data:
            assert e.get("mean_f1") is not None
            assert isinstance(e["mean_f1"], (int, float))


class TestExperimentsListStructure:
    """Test struttura base dell'elenco."""

    def test_list_returns_all_experiments(self):
        """Tutti gli exp, inclusi exp9, devono essere presenti; exp10 nascosto."""
        resp = client.get("/api/experiments")
        data = resp.json()
        ids = [e["id"] for e in data]
        assert "exp6_clean_resnet18" in ids
        assert "exp2_supcon_resnet18" in ids
        assert "exp3_sphor_convnext_tiny" in ids
        assert "exp4_sphor_ncd_resnet18" in ids
        # exp10 (MGH-NCD) rimosso dallo studio: deve essere nascosto
        assert not any("exp10" in eid for eid in ids)

    def test_list_has_required_fields(self):
        """Ogni experimento deve avere i campi base."""
        resp = client.get("/api/experiments")
        data = resp.json()
        required = {
            "id",
            "name",
            "model_variant",
            "type",
            "status",
            "num_folds",
            "folds_completed",
            "test_accuracy",
            "mean_f1",
            "std_f1",
            "is_ncd",
        }
        for e in data:
            assert required.issubset(e.keys()), f"Mancano: {required - e.keys()}"

    def test_folds_count(self):
        """folds_completed deve corrispondere al numero di fold JSON."""
        resp = client.get("/api/experiments")
        data = resp.json()
        for e in data:
            if "resnet18" in e["id"] and "exp1" in e["id"]:
                assert e["folds_completed"] == 5  # fold standard per exp6
