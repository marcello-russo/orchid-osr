"""Test per l'endpoint GET /api/embeddings.

Verifica:
- Lista embeddings con dati finti
- Lista embeddings vuota (nessun embedding presente)
- Filtro per experiment_id
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _patch_results_dir(tmp_path, monkeypatch):
    """Punta settings.results_dir su tmp_path per evitare effetti collaterali."""
    monkeypatch.setattr(settings, "results_dir", str(tmp_path))
    from app.api.routes.embeddings import _ALLOWED_EXPERIMENTS

    _ALLOWED_EXPERIMENTS.clear()


def _make_embeddings(base: Path, exp_name: str, *, seed: int = 42, n_points: int = 10):
    """Crea una struttura embeddings finta per un experimento."""
    d = base / exp_name / "embeddings"
    d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    coords = rng.randn(n_points, 3).astype(np.float32)
    np.save(str(d / "tsne_3d.npy"), coords)

    points = [
        {
            "idx": i,
            "species": f"species_{i % 3}",
            "filename": f"img_{i:04d}.jpg",
            "split": "train",
        }
        for i in range(n_points)
    ]
    (d / "metadata.json").write_text(json.dumps({"points": points}))
    return d


class TestEmbeddings:
    def test_list_embeddings_with_data(self, client, user_token):
        _make_embeddings(Path(settings.results_dir), "exp6_clean_resnet18", n_points=10)

        resp = client.get("/api/embeddings", headers={"Authorization": f"Bearer {user_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_points"] == 10
        assert len(data["points"]) == 10

    def test_list_embeddings_empty(self, client, user_token):
        resp = client.get("/api/embeddings", headers={"Authorization": f"Bearer {user_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_points"] == 0
        assert data["points"] == []

    def test_embeddings_filter_by_experiment(self, client, user_token):
        _make_embeddings(Path(settings.results_dir), "exp6_clean_resnet18", seed=42, n_points=5)
        _make_embeddings(Path(settings.results_dir), "exp2_supcon_resnet18", seed=99, n_points=8)

        resp = client.get(
            "/api/embeddings",
            params={"experiment": "exp6_clean_resnet18"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_points"] == 5
        assert len(data["points"]) == 5
        assert data["experiment"] == "exp6_clean_resnet18"
