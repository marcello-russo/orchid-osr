# tests/test_api/test_api_weights_cache.py
"""Pesi e cache dei modelli: la parte che tocca il filesystem e HF.

Sono i moduli con la copertura più bassa di tutta l'API (`weights.py`,
`cache.py`) e sono anche quelli dove un errore costa caro: un `model_id` non
validato è un path traversal, una cache senza limite è RAM esaurita, un npz
mancante è un 500 sulla classificazione. Qui si testano i confini.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.api import cache as cache_mod
from app.api import weights as weights_mod
from app.config import settings


@pytest.fixture(autouse=True)
def _clean_model_cache():
    """La cache è globale di processo: ogni test deve partire da zero."""
    cache_mod._model_cache.clear()
    cache_mod._osr_stats_cache.clear()
    yield
    cache_mod._model_cache.clear()
    cache_mod._osr_stats_cache.clear()


@pytest.fixture
def experiments(tmp_path, monkeypatch):
    """`results_dir` e la cache dei pesi puntano a directory temporanee."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    weights_cache = tmp_path / "weights-cache"

    def _cache_dir() -> Path:
        weights_cache.mkdir(parents=True, exist_ok=True)
        return weights_cache

    monkeypatch.setattr(settings, "results_dir", str(exp_dir))
    monkeypatch.setattr(weights_mod, "_get_cache_dir", _cache_dir)
    return exp_dir


def _make_experiment(
    root: Path,
    name: str,
    *,
    best_fold: int | None = 0,
    variant: str | None = None,
    mean_f1: float = 0.9,
    num_classes: int = 6,
    with_pt: bool = True,
    results: dict | None = None,
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if variant is None:
        # Come nella realtà: il variant è nel nome della directory e in results.json.
        variant = next(
            (
                v
                for v in ("dinov2_base", "dinov2_small", "convnext_small", "convnext_tiny", "resnet50", "resnet18")
                if v in name
            ),
            "resnet18",
        )
    payload = {
        "model_variant": variant,
        "best_fold": best_fold,
        "mean_f1": mean_f1,
        "std_f1": 0.01,
        "num_folds": 5,
        "test_metrics": {"accuracy": 0.91, "f1_macro": mean_f1},
    }
    if results is not None:
        payload = results
    (d / "results.json").write_text(json.dumps(payload))
    (d / "config.json").write_text(json.dumps({"num_classes": num_classes}))
    if with_pt and best_fold is not None:
        (d / f"best_fold_{best_fold}.pt").write_bytes(b"fake-pt")
    return d


class TestModelIdValidation:
    @pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "a\\b", "..", "exp1_ce/../../x"])
    def test_traversal_is_rejected(self, bad):
        with pytest.raises(ValueError, match="Invalid model_id"):
            weights_mod._validate_model_id(bad)

    def test_legitimate_id_passes(self):
        assert weights_mod._validate_model_id("exp3_sphor_resnet18") == "exp3_sphor_resnet18"


class TestSafeJsonLoad:
    def test_missing_file_is_none(self, tmp_path):
        assert weights_mod._safe_json_load(tmp_path / "nope.json") is None

    def test_malformed_json_is_none(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert weights_mod._safe_json_load(bad) is None

    def test_valid_json_is_dict(self, tmp_path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps({"a": 1}))
        assert weights_mod._safe_json_load(good) == {"a": 1}


class TestDiscoveryHelpers:
    def test_mean_f1_prefers_new_key(self):
        assert weights_mod._get_mean_f1({"mean_f1": 0.9, "test_metrics": {"f1_macro": 0.5}}) == 0.9

    def test_mean_f1_falls_back_to_legacy_test_metrics(self):
        assert weights_mod._get_mean_f1({"test_metrics": {"f1_macro": 0.5}}) == 0.5

    def test_mean_f1_absent(self):
        assert weights_mod._get_mean_f1({}) is None

    @pytest.mark.parametrize(
        ("dir_name", "num"),
        [("exp1_ce_resnet18", 1), ("exp4_sphor_ncd_dinov2_small", 4), ("exp5_sphor_ncd_x", 5), ("other", None)],
    )
    def test_exp_number(self, dir_name, num):
        assert weights_mod._get_exp_num(dir_name) == num

    def test_model_name_uses_studio_label(self):
        assert weights_mod._format_model_name("exp2_supcon_convnext_small", "convnext_small") == (
            "ConvNeXt-Small — Studio I-B (SupCon)"
        )

    def test_model_name_without_known_prefix(self):
        assert weights_mod._format_model_name("weird", "resnet18") == "ResNet-18"

    @pytest.mark.parametrize(
        ("dir_name", "size"),
        [("x_dinov2_base", 320), ("x_dinov2_small", 160), ("x_convnext_tiny", 110), ("x_resnet18", 45), ("x", 100)],
    )
    def test_estimated_size(self, dir_name, size):
        assert weights_mod._estimate_pt_size(dir_name) == size


class TestDiscoverModels:
    def test_empty_or_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "results_dir", str(tmp_path / "missing"))
        assert weights_mod._discover_models() == []

    def test_serves_only_six_class_models_above_threshold(self, experiments):
        _make_experiment(experiments, "exp1_ce_resnet18", mean_f1=0.95)
        _make_experiment(experiments, "exp2_supcon_convnext_tiny", mean_f1=0.85)
        _make_experiment(experiments, "exp3_sphor_resnet50", mean_f1=0.60)  # sotto soglia
        _make_experiment(experiments, "exp4_sphor_ncd_dinov2_small", num_classes=5)  # testa non-6
        _make_experiment(experiments, "not_an_experiment")  # prefisso sconosciuto

        models = weights_mod._discover_models()
        assert [m.id for m in models] == ["exp2_supcon_convnext_tiny", "exp1_ce_resnet18"]

    def test_fields_come_from_results_json(self, experiments):
        _make_experiment(experiments, "exp3_sphor_resnet18", best_fold=2, mean_f1=0.93)
        (m,) = weights_mod._discover_models()
        assert m.id == "exp3_sphor_resnet18"
        assert m.model_variant == "resnet18"
        assert m.mean_f1 == 0.93
        assert m.test_accuracy == 0.91
        assert m.num_folds == 5
        assert m.exp_num == 3
        assert m.is_cached is True
        assert m.tags == ["best_f1", "ncd"] or set(m.tags) >= {"best_f1"}

    def test_missing_pt_is_reported_as_not_cached(self, experiments):
        _make_experiment(experiments, "exp1_ce_resnet18", with_pt=False)
        (m,) = weights_mod._discover_models()
        assert m.is_cached is False

    def test_results_without_best_fold_are_skipped(self, experiments):
        _make_experiment(experiments, "exp1_ce_resnet18", best_fold=None)
        assert weights_mod._discover_models() == []

    def test_cache_status_sums_sizes(self, experiments):
        _make_experiment(experiments, "exp1_ce_resnet18")  # 45 MB
        _make_experiment(experiments, "exp2_supcon_resnet50")  # 100 MB
        status = weights_mod.get_cache_status()
        assert status.total_models == 2
        assert status.cached_models == 2
        assert status.total_size_mb == 145
        assert status.cached_size_mb == 145
        assert status.is_downloading is False


class TestResolveWeights:
    def test_local_checkpoint_wins(self, experiments):
        d = _make_experiment(experiments, "exp1_ce_resnet18", best_fold=1)
        path, variant = weights_mod._resolve_weights("exp1_ce_resnet18")
        assert path == d / "best_fold_1.pt"
        assert variant == "resnet18"

    def test_falls_back_to_cache(self, experiments):
        d = _make_experiment(experiments, "exp1_ce_resnet18", best_fold=0, with_pt=False)
        cached = weights_mod._get_cache_dir() / "exp1_ce_resnet18_best_fold_0.pt"
        cached.write_bytes(b"cached")
        path, _ = weights_mod._resolve_weights("exp1_ce_resnet18")
        assert path == cached
        assert not (d / "best_fold_0.pt").exists()

    def test_unknown_experiment_is_none(self, experiments):
        assert weights_mod._resolve_weights("exp1_ce_nope") is None

    def test_missing_best_fold_is_none(self, experiments):
        _make_experiment(experiments, "exp1_ce_resnet18", best_fold=None)
        assert weights_mod._resolve_weights("exp1_ce_resnet18") is None

    def test_corrupt_results_is_none(self, experiments):
        d = experiments / "exp1_ce_resnet18"
        d.mkdir()
        (d / "results.json").write_text("{broken")
        assert weights_mod._resolve_weights("exp1_ce_resnet18") is None

    def test_download_failure_returns_none(self, experiments, monkeypatch):
        _make_experiment(experiments, "exp1_ce_resnet18", with_pt=False)

        def boom(**_kwargs):
            raise OSError("hub unreachable")

        monkeypatch.setattr(weights_mod, "hf_hub_download", boom)
        assert weights_mod._resolve_weights("exp1_ce_resnet18") is None


class TestFindWeights:
    def test_picks_highest_f1_for_variant(self, experiments):
        _make_experiment(experiments, "exp1_ce_resnet18", best_fold=0, mean_f1=0.88)
        best = _make_experiment(experiments, "exp3_sphor_resnet18", best_fold=1, mean_f1=0.94)
        _make_experiment(experiments, "exp2_supcon_resnet50", best_fold=0, mean_f1=0.99)
        assert weights_mod._find_weights("resnet18") == best / "best_fold_1.pt"

    def test_ignores_non_six_class_heads(self, experiments):
        _make_experiment(experiments, "exp4_sphor_ncd_resnet18", best_fold=0, mean_f1=0.99, num_classes=5)
        assert weights_mod._find_weights("resnet18") is None

    def test_nothing_found(self, experiments):
        assert weights_mod._find_weights("dinov2_base") is None


class _FakeModel:
    """Sostituto di ClassifierModel: registra cosa gli viene chiesto."""

    def __init__(self, variant: str):
        self.variant = variant
        self.loaded_from: str | None = None
        self.eval_called = False
        self.device = None

    def load_weights(self, path: str) -> None:
        self.loaded_from = path

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self


class TestGetModel:
    @pytest.fixture
    def fake_backend(self, monkeypatch):
        created: list[_FakeModel] = []

        def create_model(variant, **_kwargs):
            m = _FakeModel(variant)
            created.append(m)
            return m

        monkeypatch.setattr("app.core.models.create_model", create_model)
        monkeypatch.setattr(cache_mod, "get_device", lambda: "cpu")
        return created

    def test_explicit_model_id_uses_resolved_weights(self, experiments, fake_backend):
        _make_experiment(experiments, "exp2_supcon_convnext_tiny", best_fold=0)
        model = cache_mod.get_model(model_id="exp2_supcon_convnext_tiny")
        assert model.variant == "convnext_tiny"
        assert model.loaded_from.endswith("best_fold_0.pt")
        assert model.eval_called and model.device == "cpu"

    def test_cached_model_is_reused(self, experiments, fake_backend):
        _make_experiment(experiments, "exp1_ce_resnet18")
        first = cache_mod.get_model(model_id="exp1_ce_resnet18")
        second = cache_mod.get_model(model_id="exp1_ce_resnet18")
        assert first is second
        assert len(fake_backend) == 1

    def test_cache_evicts_oldest_at_limit(self, experiments, fake_backend):
        assert cache_mod.MAX_CACHED_MODELS == 2
        for name in ("exp1_ce_resnet18", "exp2_supcon_resnet50", "exp3_sphor_convnext_tiny"):
            _make_experiment(experiments, name)
        a = cache_mod.get_model(model_id="exp1_ce_resnet18")
        cache_mod.get_model(model_id="exp2_supcon_resnet50")
        assert len(cache_mod._model_cache) == 2
        # ricarico a: sposta a in coda, poi entro con c → esce b (il più vecchio)
        assert cache_mod.get_model(model_id="exp1_ce_resnet18") is a
        cache_mod.get_model(model_id="exp3_sphor_convnext_tiny")
        keys = list(cache_mod._model_cache)
        assert keys[-1].startswith("exp3_sphor_convnext_tiny")
        assert any(k.startswith("exp1_ce_resnet18") for k in keys)
        assert not any(k.startswith("exp2_supcon_resnet50") for k in keys)

    def test_unknown_model_type_raises(self, experiments, fake_backend):
        with pytest.raises(ValueError, match="Unknown model type"):
            cache_mod.get_model(model_type="not_a_model")

    def test_unknown_model_id_raises(self, experiments, fake_backend):
        with pytest.raises(ValueError, match="No weights found"):
            cache_mod.get_model(model_id="exp1_ce_missing")


class TestGetOsrStats:
    @pytest.fixture
    def stats_file(self, tmp_path, monkeypatch):
        """npz valido + directory di cache temporanea."""
        stats_dir = tmp_path / "osr-stats"
        monkeypatch.setattr(cache_mod, "_STATS_CACHE_DIR", stats_dir)
        monkeypatch.setattr(cache_mod, "hf_exp_dir", lambda model_id: f"remote/{model_id}")

        rng = np.random.default_rng(0)
        data = {
            "train_embeddings": rng.normal(size=(10, 4)).astype("float32"),
            "class_means": rng.normal(size=(3, 4)).astype("float32"),
            "cov_inv": np.eye(4, dtype="float32"),
            "threshold_ensemble": np.array([0.42], dtype="float32"),
            "maha_mean": np.array([1.5], dtype="float32"),
            "maha_std": np.array([0.25], dtype="float32"),
            "knn_mean": np.array([2.5], dtype="float32"),
            "knn_std": np.array([0.5], dtype="float32"),
        }
        return stats_dir, data

    def test_loads_npz_and_caches_it(self, stats_file, monkeypatch):
        stats_dir, data = stats_file
        stats_dir.mkdir(parents=True)
        np.savez(stats_dir / "exp3_sphor_resnet18.npz", **data)

        stats = cache_mod.get_osr_stats("exp3_sphor_resnet18")
        assert stats["threshold_ensemble"] == pytest.approx(0.42)
        assert stats["maha_mean"] == pytest.approx(1.5)
        assert stats["knn_std"] == pytest.approx(0.5)
        assert stats["nn"].n_neighbors == 5
        assert stats["class_means"].shape == (3, 4)

        # secondo giro: dalla cache in memoria, senza rileggere il file
        (stats_dir / "exp3_sphor_resnet18.npz").unlink()
        assert cache_mod.get_osr_stats("exp3_sphor_resnet18") is stats

    def test_unknown_experiment_prefix_is_none(self, monkeypatch):
        monkeypatch.setattr(cache_mod, "hf_exp_dir", lambda model_id: None)
        assert cache_mod.get_osr_stats("weird_id") is None

    def test_download_failure_is_none(self, stats_file, monkeypatch):
        def boom(**_kwargs):
            raise OSError("no network")

        monkeypatch.setattr(cache_mod, "hf_hub_download", boom)
        assert cache_mod.get_osr_stats("exp3_sphor_resnet18") is None

    def test_missing_optional_keys_get_defaults(self, stats_file):
        stats_dir, data = stats_file
        stats_dir.mkdir(parents=True)
        minimal = {k: data[k] for k in ("train_embeddings", "class_means", "cov_inv", "threshold_ensemble")}
        np.savez(stats_dir / "minimal.npz", **minimal)

        stats = cache_mod.get_osr_stats("minimal")
        assert stats["maha_mean"] == 0.0
        assert stats["maha_std"] == 1.0
        assert stats["knn_mean"] == 0.0
        assert stats["knn_std"] == 1.0

    def test_corrupt_npz_is_none(self, stats_file):
        stats_dir, _ = stats_file
        stats_dir.mkdir(parents=True)
        (stats_dir / "broken.npz").write_bytes(b"not a npz")
        assert cache_mod.get_osr_stats("broken") is None
