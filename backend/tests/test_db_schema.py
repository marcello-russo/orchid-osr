"""Test schema SQLite."""

import os
import sqlite3
import tempfile

from app.core.db import _init_schema


def test_schema_creates_tables():
    db = tempfile.mktemp(suffix=".db")
    try:
        conn = sqlite3.connect(db)
        _init_schema(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        names = [r[0] for r in tables]
        assert "experiments" in names
        assert "folds" in names
        assert "users" in names
        assert "invites" in names
        assert len(names) == 4
    finally:
        os.unlink(db)


def test_schema_columns_experiments():
    db = tempfile.mktemp(suffix=".db")
    try:
        conn = sqlite3.connect(db)
        _init_schema(conn)
        cols = conn.execute("PRAGMA table_info(experiments)").fetchall()
        col_names = {r[1] for r in cols}
        for required in ["id", "model_variant", "mean_f1", "mtime"]:
            assert required in col_names
    finally:
        os.unlink(db)


def test_schema_idempotent():
    db = tempfile.mktemp(suffix=".db")
    try:
        conn = sqlite3.connect(db)
        _init_schema(conn)
        _init_schema(conn)
        tables = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
        assert tables == 4
    finally:
        os.unlink(db)
