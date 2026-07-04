"""Test sync functions."""

import json
import sqlite3

from app.core.db import _init_schema, sync_experiment


def _mem_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_sync_experiment_creates_row(tmp_path):
    exp_dir = tmp_path / "exp_test_resnet18"
    exp_dir.mkdir()
    with open(exp_dir / "config.json", "w") as f:
        json.dump({"model_variant": "resnet18", "num_classes": 6}, f)
    with open(exp_dir / "results.json", "w") as f:
        json.dump({"mean_f1": 0.85, "best_fold": 0}, f)
    with open(exp_dir / "fold_0.json", "w") as f:
        json.dump({"best_f1": 0.88, "fold": 0}, f)

    conn = _mem_db()
    _init_schema(conn)
    sync_experiment(str(exp_dir), conn)

    row = conn.execute("SELECT * FROM experiments").fetchone()
    assert row is not None
    assert row["model_variant"] == "resnet18"
    assert row["mean_f1"] == 0.85
    assert row["started_at"] is not None
    assert row["completed_at"] is not None


def test_sync_experiment_creates_folds(tmp_path):
    exp_dir = tmp_path / "exp_test_resnet18"
    exp_dir.mkdir()
    with open(exp_dir / "config.json", "w") as f:
        json.dump({"model_variant": "resnet18", "num_classes": 6}, f)
    with open(exp_dir / "results.json", "w") as f:
        json.dump({"mean_f1": 0.85}, f)
    for i in range(3):
        with open(exp_dir / f"fold_{i}.json", "w") as f:
            json.dump({"best_f1": 0.8 + i * 0.05, "fold": i}, f)

    conn = _mem_db()
    _init_schema(conn)
    sync_experiment(str(exp_dir), conn)

    folds = conn.execute("SELECT * FROM folds ORDER BY fold_idx").fetchall()
    assert len(folds) == 3
    assert folds[0]["best_f1"] == 0.8


def test_sync_idempotent(tmp_path):
    """Sync ripetuto non crea duplicati grazie a mtime."""
    exp_dir = tmp_path / "exp_test_resnet18"
    exp_dir.mkdir()
    with open(exp_dir / "config.json", "w") as f:
        json.dump({"model_variant": "resnet18", "num_classes": 6}, f)
    with open(exp_dir / "results.json", "w") as f:
        json.dump({"mean_f1": 0.85}, f)

    conn = _mem_db()
    _init_schema(conn)
    sync_experiment(str(exp_dir), conn)
    sync_experiment(str(exp_dir), conn)

    count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 1


def test_sync_skips_if_mtime_unchanged(tmp_path):
    """Se mtime non cambia, sync non fa nulla (idempotenza mtime)."""
    exp_dir = tmp_path / "exp_test_resnet18"
    exp_dir.mkdir()
    with open(exp_dir / "config.json", "w") as f:
        json.dump({"model_variant": "resnet18", "num_classes": 6}, f)
    with open(exp_dir / "results.json", "w") as f:
        json.dump({"mean_f1": 0.85}, f)

    conn = _mem_db()
    _init_schema(conn)
    sync_experiment(str(exp_dir), conn)
    mtime_1 = conn.execute("SELECT mtime FROM experiments").fetchone()["mtime"]

    # Second sync senza modifiche ai file
    sync_experiment(str(exp_dir), conn)
    mtime_2 = conn.execute("SELECT mtime FROM experiments").fetchone()["mtime"]

    assert mtime_1 == mtime_2
    assert mtime_1 > 0


def test_sync_updated_if_mtime_changes(tmp_path):
    """Se results.json viene modificato, sync aggiorna il DB."""
    exp_dir = tmp_path / "exp_test_resnet18"
    exp_dir.mkdir()
    with open(exp_dir / "config.json", "w") as f:
        json.dump({"model_variant": "resnet18", "num_classes": 6}, f)
    with open(exp_dir / "results.json", "w") as f:
        json.dump({"mean_f1": 0.85}, f)

    conn = _mem_db()
    _init_schema(conn)
    sync_experiment(str(exp_dir), conn)

    # Modifica results.json (aggiorna mtime)
    import time

    time.sleep(0.1)
    with open(exp_dir / "results.json", "w") as f:
        json.dump({"mean_f1": 0.92}, f)

    sync_experiment(str(exp_dir), conn)
    row = conn.execute("SELECT mean_f1 FROM experiments").fetchone()
    assert row["mean_f1"] == 0.92
