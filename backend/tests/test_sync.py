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
