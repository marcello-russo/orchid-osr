"""Test schema SQLite."""

import sqlite3

import pytest

from app.core.db import _init_schema

EXPECTED_TABLES = {
    "auth_meta",
    "experiments",
    "folds",
    "guest_passes",
    "invites",
    "login_attempts",
    "users",
}


def _open(tmp_path):
    conn = sqlite3.connect(tmp_path / "orchid.db")
    _init_schema(conn)
    return conn


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}


def test_schema_creates_tables(tmp_path):
    conn = _open(tmp_path)
    assert _tables(conn) == EXPECTED_TABLES


def test_schema_columns_experiments(tmp_path):
    conn = _open(tmp_path)
    col_names = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
    for required in ["id", "model_variant", "mean_f1", "mtime"]:
        assert required in col_names


def test_superadmin_role_is_accepted(tmp_path):
    conn = _open(tmp_path)
    conn.execute("INSERT INTO users (username, password_hash, role) VALUES ('s', 'h', 'superadmin')")
    assert conn.execute("SELECT role FROM users WHERE username='s'").fetchone()[0] == "superadmin"


def test_legacy_auth_tables_are_migrated_without_losing_rows(tmp_path):
    path = tmp_path / "orchid.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'admin')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE invites (
            code TEXT PRIMARY KEY,
            used INTEGER NOT NULL DEFAULT 0 CHECK(used IN (0, 1)),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO users (username, password_hash, role) VALUES ('vecchio', 'hash', 'admin');
        INSERT INTO invites (code, used) VALUES ('INV-OLD', 1);
        """
    )
    legacy.commit()
    legacy.close()

    conn = sqlite3.connect(path)
    _init_schema(conn)

    row = conn.execute("SELECT role FROM users WHERE username='vecchio'").fetchone()
    assert row[0] == "admin"
    inv = conn.execute("SELECT role, max_uses, uses, used FROM invites WHERE code='INV-OLD'").fetchone()
    assert tuple(inv) == ("user", 1, 1, 1)
    # la tabella nuova accetta il ruolo che quella vecchia rifiutava
    conn.execute("INSERT INTO users (username, password_hash, role) VALUES ('nuovo', 'h', 'superadmin')")


def test_schema_idempotent(tmp_path):
    conn = _open(tmp_path)
    _init_schema(conn)
    assert _tables(conn) == EXPECTED_TABLES


def test_db_path_without_directory(tmp_path, monkeypatch):
    from app.core import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", "orchid-test.db")
    monkeypatch.setattr(db_module, "_SCHEMA_READY", set())
    monkeypatch.chdir(tmp_path)
    conn = db_module.get_db()
    conn.close()
    assert (tmp_path / "orchid-test.db").exists()


@pytest.mark.parametrize("table", ["users", "invites", "guest_passes"])
def test_auth_tables_have_expected_columns(tmp_path, table):
    conn = _open(tmp_path)
    columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    assert columns
