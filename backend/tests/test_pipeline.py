"""Tests for pipeline.py — NCD pipeline orchestrator."""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from app.core.ncd.extraction import ExtractionBundle


def test_run_ncd_pipeline_validates_params():
    """run_ncd_pipeline fa fail-fast su parametri mancanti."""
    from app.core.ncd.pipeline import run_ncd_pipeline

    result = run_ncd_pipeline(
        exp_dir="/tmp/fake",
        model_variant="resnet18",
        num_classes=6,
        class_names=["A", "B", "C", "D", "E", "F"],
        data_dir="/tmp/fake_train",
        run_osr=True,
        ncd_mode=True,
    )
    assert result is None  # perche' manca all_classes


def test_run_ncd_pipeline_no_test_dir():
    """run_ncd_pipeline fallisce senza test_dir in OSR mode."""
    from app.core.ncd.pipeline import run_ncd_pipeline

    result = run_ncd_pipeline(
        exp_dir="/tmp/fake",
        model_variant="resnet18",
        num_classes=6,
        class_names=["A", "B", "C", "D", "E", "F"],
        data_dir="/tmp/fake_train",
        run_osr=True,
        test_dir=None,
    )
    assert result is None


def test_run_ncd_pipeline_missing_model():
    """run_ncd_pipeline fallisce se il checkpoint non esiste."""
    from app.core.ncd.pipeline import run_ncd_pipeline

    result = run_ncd_pipeline(
        exp_dir="/tmp/fake_exp_12345",
        model_variant="resnet18",
        num_classes=6,
        class_names=["A", "B", "C", "D", "E", "F"],
        data_dir="/tmp/fake_train",
        run_osr=True,
        test_dir="/tmp/fake_test",
        ncd_mode=True,
        all_classes=["A", "B", "C", "D", "E", "F", "G"],
    )
    assert result is None


def test_run_ncd_pipeline_with_mocks():
    """Pipeline completa con extraction mockata — clustering + detector."""
    from app.core.ncd.pipeline import run_ncd_pipeline

    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "results.json"), "w") as f:
        json.dump({"best_fold": 0}, f)

    # Salva un checkpoint finto (solo pesi di prova)
    dummy = torch.nn.Linear(8, 3)
    torch.save(dummy.state_dict(), os.path.join(tmpdir, "best_fold_0.pt"))
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump({"model_variant": "resnet18"}, f)

    # Dati sintetici
    n_train, n_test, d = 30, 15, 8
    rng = np.random.default_rng(42)
    bundle = ExtractionBundle(
        train_embeddings=rng.random((n_train, d)).astype(np.float32),
        train_logits=rng.random((n_train, 3)).astype(np.float32),
        train_labels=np.zeros(n_train, dtype=np.int64),
        train_class_names=["A", "B", "C"],
        test_embeddings=rng.random((n_test, d)).astype(np.float32),
        test_logits=rng.random((n_test, 3)).astype(np.float32),
        test_labels=np.zeros(n_test, dtype=np.int64),
        test_class_names=["A", "B", "C"],
    )

    with patch("app.core.ncd.pipeline.extract_all", return_value=bundle):
        result = run_ncd_pipeline(
            exp_dir=tmpdir,
            model_variant="resnet18",
            num_classes=3,
            class_names=["A", "B", "C"],
            data_dir="/tmp/fake_train",
            batch_size=8,
            run_osr=True,
            test_dir="/tmp/fake_test",
            run_dbscan=False,
            run_spectral=False,
        )

    assert result is not None
    assert "clustering" in result
    assert "kmeans_best" in result["clustering"]
    assert "detectors" in result
