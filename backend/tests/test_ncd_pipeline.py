"""Integration-style tests for NCD pipeline parameter validation and wiring."""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from app.core.models import create_model
from app.core.ncd.pipeline import run_ncd_pipeline


class TestNcdPipelineValidation:
    """Verify that run_ncd_pipeline validates its parameters correctly."""

    def test_ncd_mode_without_all_classes_returns_none(self):
        """ncd_mode=True senza all_classes deve fallire con log error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_ncd_pipeline(
                exp_dir=tmpdir,
                model_variant="resnet18",
                num_classes=6,
                class_names=["a", "b", "c", "d", "e", "f"],
                data_dir=tmpdir,
                batch_size=4,
                run_osr=True,
                ncd_mode=True,
                all_classes=None,
            )
            assert result is None

    def test_osr_without_test_dir_returns_none(self):
        """run_osr=True senza test_dir deve fallire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_ncd_pipeline(
                exp_dir=tmpdir,
                model_variant="resnet18",
                num_classes=6,
                class_names=["a", "b", "c", "d", "e", "f"],
                data_dir=tmpdir,
                batch_size=4,
                run_osr=True,
                test_dir=None,
            )
            assert result is None

    def test_ncd_mode_with_all_classes_passes_initial_validation(self):
        """Config valida supera la validazione parametri iniziale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            emb_dir = os.path.join(tmpdir, "embeddings")
            os.makedirs(emb_dir, exist_ok=True)
            np.save(os.path.join(emb_dir, "embeddings.npy"), np.random.rand(20, 10))
            np.save(os.path.join(emb_dir, "labels.npy"), np.array([0] * 10 + [1] * 10))

            with (
                patch("app.core.ncd.clustering.grid_search_kmeans") as mock_grid,
                patch("app.core.ncd.clustering.kmeans_clustering") as mock_km,
                patch("app.core.ncd.clustering.dbscan_clustering") as mock_db,
                patch("app.core.ncd.clustering.spectral_clustering") as mock_sp,
                patch("app.core.ncd.evaluation.compute_all_metrics") as mock_metrics,
                patch("app.core.ncd.pipeline.logger"),
            ):
                mock_grid.return_value = {
                    "k": 3,
                    "acc": 0.5,
                    "nmi": 0.3,
                    "ari": 0.2,
                    "ami": 0.2,
                    "f1": 0.4,
                    "silhouette": 0.1,
                }  # noqa: E501
                mock_km.return_value = (np.zeros(20), {"n_clusters": 3, "inertia": 1.0})
                mock_db.return_value = (np.zeros(20), {"n_clusters": 3, "noise_ratio": 0.05})
                mock_sp.return_value = (np.zeros(20), {"n_clusters": 3})
                mock_metrics.return_value = {
                    "acc": 0.5,
                    "nmi": 0.3,
                    "ari": 0.2,
                    "ami": 0.2,
                    "f_measure": 0.4,
                    "silhouette": 0.1,
                }  # noqa: E501

                result = run_ncd_pipeline(
                    exp_dir=tmpdir,
                    model_variant="resnet18",
                    num_classes=6,
                    class_names=["a", "b", "c", "d", "e", "f"],
                    data_dir=tmpdir,
                    batch_size=4,
                    run_osr=True,
                    test_dir=tmpdir,
                    ncd_mode=True,
                    all_classes=["a", "b", "c", "d", "e", "f", "g"],
                )
                assert result is None

    def _prepare_ncd_env(self, tmpdir):
        """Set up temporary environment for full NCD pipeline test."""
        emb_dir = os.path.join(tmpdir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        np.save(os.path.join(emb_dir, "embeddings.npy"), np.random.rand(20, 10))
        np.save(os.path.join(emb_dir, "labels.npy"), np.array([0] * 10 + [1] * 10))
        results = {"best_fold": 0}
        with open(os.path.join(tmpdir, "results.json"), "w") as f:
            json.dump(results, f)
        # Must match model architecture expected by load_trained_model
        model = create_model("resnet18", num_classes=6, pretrained=False)
        torch.save(model.state_dict(), os.path.join(tmpdir, "best_fold_0.pt"))

    def test_openset_results_structure(self):
        """Verifica struttura risultati OSR con tutti i moduli mockati."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._prepare_ncd_env(tmpdir)

            from types import SimpleNamespace

            with (
                patch("app.core.ncd.pipeline.extract_all") as mock_extract_all,
                patch("app.core.ncd.clustering.grid_search_kmeans") as mock_grid,
                patch("app.core.ncd.clustering.kmeans_clustering") as mock_km,
                patch("app.core.ncd.clustering.dbscan_clustering") as mock_db,
                patch("app.core.ncd.clustering.spectral_clustering") as mock_sp,
                patch("app.core.ncd.evaluation.compute_all_metrics") as mock_metrics,
                patch("sklearn.manifold.TSNE") as mock_tsne,
                patch("app.core.ncd.pipeline.logger"),
            ):
                mock_extract_all.return_value = SimpleNamespace(
                    train_embeddings=np.random.rand(10, 10),
                    train_logits=None,
                    train_labels=np.array([0] * 5 + [1] * 5),
                    train_class_names=["a", "b", "c", "d", "e", "f"],
                    test_embeddings=np.random.rand(10, 10),
                    test_logits=None,
                    test_labels=np.array([0] * 5 + [6] * 5),
                    test_class_names=["a", "b", "c", "d", "e", "f", "g"],
                )
                mock_grid.return_value = {
                    "k": 3,
                    "acc": 0.5,
                    "nmi": 0.3,
                    "ari": 0.2,
                    "ami": 0.2,
                    "f1": 0.4,
                    "silhouette": 0.1,
                }  # noqa: E501
                mock_km.return_value = (np.zeros(20), {"n_clusters": 3, "inertia": 1.0})
                mock_db.return_value = (np.zeros(20), {"n_clusters": 3, "noise_ratio": 0.05})
                mock_sp.return_value = (np.zeros(20), {"n_clusters": 3})
                mock_metrics.return_value = {
                    "acc": 0.5,
                    "nmi": 0.3,
                    "ari": 0.2,
                    "ami": 0.2,
                    "f_measure": 0.4,
                    "silhouette": 0.1,
                }  # noqa: E501
                mock_tsne.return_value.fit_transform.return_value = np.random.rand(20, 2)

                result = run_ncd_pipeline(
                    exp_dir=tmpdir,
                    model_variant="resnet18",
                    num_classes=6,
                    class_names=["a", "b", "c", "d", "e", "f"],
                    data_dir=tmpdir,
                    batch_size=4,
                    run_osr=False,
                    test_dir=tmpdir,
                    ncd_mode=True,
                    all_classes=["a", "b", "c", "d", "e", "f", "g"],
                )
                assert isinstance(result, dict), "Dovrebbe tornare un dict"
                assert "clustering" in result


_GPU_WEIGHTS = "experiments/exp5_sphor_ncd_resnet18/best_fold_1.pt"
_needs_gpu_data = not (torch.cuda.is_available() and os.path.exists(_GPU_WEIGHTS))


@pytest.mark.skipif(_needs_gpu_data, reason="requires CUDA + Exp5 resnet18 weights + dataset (GPU machine only)")
def test_ncd_pipeline_saves_osr_scores():
    """run_ncd_pipeline must write osr_scores.npz with y_true + per-detector scores.

    GPU-only (loads best_fold_1.pt 43M + CUDA, ~3 min). Uses the real Exp5
    resnet18 dir because the pipeline resolves the checkpoint from
    <exp_dir>/results.json; a tmp dir would exit early with no model.
    Skipped (not failed) on CI without data.
    """
    import numpy as np

    from app.core.ncd import pipeline as pipe

    exp_dir = "experiments/exp5_sphor_ncd_resnet18"
    pipe.run_ncd_pipeline(
        exp_dir=exp_dir,
        model_variant="resnet18",
        num_classes=4,
        class_names=["O. exaltata", "O. incubacea", "O. sphegodes", "O. sphegodes_Palena"],
        data_dir="dataset/cropped_256x512/train_ncd_exp5",
        test_dir="dataset/cropped_256x512/test",
        run_osr=True,
        ncd_mode=True,
        exclude_classes=["O. garganica", "O. majellensis"],
        all_classes=[
            "O. exaltata",
            "O. garganica",
            "O. incubacea",
            "O. majellensis",
            "O. sphegodes",
            "O. sphegodes_Palena",
        ],
    )
    scores_path = os.path.join(exp_dir, "osr_scores.npz")
    assert os.path.exists(scores_path)
    z = np.load(scores_path)
    assert "y_true" in z and set(np.unique(z["y_true"])) == {0, 1}
    assert int(z["y_true"].sum()) == 100
    assert len(z["y_true"]) == 300
