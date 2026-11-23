"""Test per le estensioni NCD del DB e migrazione."""

import json
import sqlite3
import time

from app.core.db import (
    _init_schema,
    _migrate_schema,
    _read_ncd_metrics,
    sync_experiment,
)


def _mem_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_ncd_columns_added_by_migration():
    """Verifica che la migrazione aggiunga le colonne NCD."""
    conn = _mem_db()
    _init_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)").fetchall()}
    for col in ("ncd_auroc", "ncd_aupr", "ncd_detector", "ncd_clustering_nmi", "is_ncd"):
        assert col in cols, f"Colonna {col} mancante dopo migrazione"


def test_migration_idempotent():
    """Migrazione eseguita due volte non causa errori."""
    conn = _mem_db()
    _init_schema(conn)
    _migrate_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)").fetchall()}
    for col in ("ncd_auroc", "ncd_detector", "is_ncd"):
        assert col in cols


def test_read_ncd_metrics_empty(tmp_path):
    """Nessun ncd_results.json → dict vuoto."""
    assert _read_ncd_metrics(tmp_path) == {}


def test_read_ncd_metrics_with_detectors(tmp_path):
    """Regola tesi: ncd_auroc = Ensemble anche se KNN è più alto."""
    ncd = {
        "detectors": {
            "Mahalanobis": {"auroc": 0.85},
            "KNN": {"auroc": 0.92},
            "Energy": {"auroc": 0.71},
            "Ensemble": {"auroc": 0.90},
        },
        "clustering": {
            "kmeans_best": {"nmi": 0.45},
            "spectral": {"nmi": 0.51},
        },
    }
    ncd_file = tmp_path / "ncd_results.json"
    ncd_file.write_text(json.dumps(ncd))

    metrics = _read_ncd_metrics(tmp_path)
    assert metrics["ncd_auroc"] == 0.90  # Ensemble, non il max (KNN 0.92)
    assert metrics["ncd_detector"] == "Ensemble"
    assert metrics["ncd_clustering_nmi"] == 0.51  # spectral is best


def test_read_ncd_metrics_no_detectors(tmp_path):
    """Solo clustering, nessun detector → solo clustering salvato."""
    ncd = {
        "clustering": {
            "kmeans_best": {"nmi": 0.35},
        },
    }
    ncd_file = tmp_path / "ncd_results.json"
    ncd_file.write_text(json.dumps(ncd))

    metrics = _read_ncd_metrics(tmp_path)
    assert "ncd_auroc" not in metrics
    assert metrics["ncd_clustering_nmi"] == 0.35


def test_read_ncd_metrics_skips_malformed(tmp_path):
    """Detector senza campo auroc viene saltato."""
    ncd = {
        "detectors": {
            "WeirdDetector": {"something_else": 0.5},
            "NormalKNN": {"auroc": 0.88},
        },
    }
    ncd_file = tmp_path / "ncd_results.json"
    ncd_file.write_text(json.dumps(ncd))

    metrics = _read_ncd_metrics(tmp_path)
    assert metrics["ncd_auroc"] == 0.88
    assert metrics["ncd_detector"] == "NormalKNN"


def test_sync_ncd_metrics(tmp_path):
    """Sync popola ncd_auroc, is_ncd e campi correlati nel DB."""
    exp_dir = tmp_path / "exp10_test_ncd"
    exp_dir.mkdir()
    cfg = {"model_variant": "dinov2_small", "num_classes": 6}
    (exp_dir / "config.json").write_text(json.dumps(cfg))
    (exp_dir / "results.json").write_text(json.dumps({"mean_f1": 0.85}))

    ncd = {
        "detectors": {"Ensemble": {"auroc": 0.91}},
        "clustering": {"kmeans_best": {"nmi": 0.48}},
    }
    (exp_dir / "ncd_results.json").write_text(json.dumps(ncd))

    conn = _mem_db()
    _init_schema(conn)
    sync_experiment(str(exp_dir), conn)

    row = conn.execute("SELECT * FROM experiments").fetchone()
    assert row["ncd_auroc"] == 0.91
    assert row["ncd_detector"] == "Ensemble"
    assert row["ncd_clustering_nmi"] == 0.48
    assert row["is_ncd"] == 0  # ncd_mode non nel config


def test_sync_ncd_marks_is_ncd_from_config(tmp_path):
    """Se config ha ncd_mode=True, is_ncd=1."""
    exp_dir = tmp_path / "exp10_test_ncd_mode"
    exp_dir.mkdir()
    cfg = {"model_variant": "dinov2_small", "num_classes": 6, "ncd_mode": True}
    (exp_dir / "config.json").write_text(json.dumps(cfg))
    (exp_dir / "results.json").write_text(json.dumps({"mean_f1": 0.85}))

    conn = _mem_db()
    _init_schema(conn)
    sync_experiment(str(exp_dir), conn)

    row = conn.execute("SELECT is_ncd FROM experiments").fetchone()
    assert row["is_ncd"] == 1


def test_sync_ncd_with_mtime_resync(tmp_path):
    """Modifica ncd_results.json → resync aggiorna DB."""
    exp_dir = tmp_path / "exp10_resync_test"
    exp_dir.mkdir()
    cfg = {"model_variant": "convnext_tiny", "num_classes": 6}
    (exp_dir / "config.json").write_text(json.dumps(cfg))
    (exp_dir / "results.json").write_text(json.dumps({"mean_f1": 0.70}))

    conn = _mem_db()
    _init_schema(conn)
    sync_experiment(str(exp_dir), conn)

    # Aggiungi ncd_results.json dopo primo sync
    (exp_dir / "ncd_results.json").write_text(json.dumps({"detectors": {"KNN": {"auroc": 0.85}}}))
    # Aggiorna mtime di results.json per forzare resync
    time.sleep(0.1)
    (exp_dir / "results.json").write_text(json.dumps({"mean_f1": 0.70}))

    sync_experiment(str(exp_dir), conn)
    row = conn.execute("SELECT ncd_auroc FROM experiments").fetchone()
    assert row["ncd_auroc"] == 0.85
