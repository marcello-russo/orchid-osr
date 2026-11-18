"""Test del layer di persistenza auth (SQLite, transazioni)."""

import json

import pytest

from app.core import auth_store
from app.core import db as db_module


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "orchid.db"))
    auth_store.reset_for_tests()
    yield tmp_path
    auth_store.reset_for_tests()


def _invite(code: str = "INV-1", **overrides):
    record = {
        "code": code,
        "role": "user",
        "used": False,
        "used_by": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "expires_at": None,
        "max_uses": 1,
        "uses": 0,
        "label": "",
        "revoked": False,
    }
    record.update(overrides)
    return record


def test_user_roundtrip(store):
    with auth_store.transaction() as conn:
        auth_store.insert_user(conn, "mario", "hash", "admin")
    assert auth_store.get_user("mario")["role"] == "admin"
    assert [u["username"] for u in auth_store.list_users()] == ["mario"]


def test_transaction_rolls_back_on_exception(store):
    with pytest.raises(RuntimeError):
        with auth_store.transaction() as conn:
            auth_store.insert_user(conn, "fantasma", "hash", "user")
            raise RuntimeError("boom")
    assert auth_store.get_user("fantasma") is None


def test_consume_invite_marks_it_used(store):
    with auth_store.transaction() as conn:
        auth_store.insert_invite(conn, _invite())
        auth_store.consume_invite(conn, "INV-1", "mario")
    inv = auth_store.get_invite("INV-1")
    assert inv["uses"] == 1 and inv["used"] is True and inv["used_by"] == "mario"


def test_consume_multiuse_stops_at_max(store):
    with auth_store.transaction() as conn:
        auth_store.insert_invite(conn, _invite(max_uses=2))
        auth_store.consume_invite(conn, "INV-1", "a")
        auth_store.consume_invite(conn, "INV-1", "b")
    inv = auth_store.get_invite("INV-1")
    assert inv["uses"] == 2 and inv["used"] is True


def test_revoke_invite_is_idempotent(store):
    with auth_store.transaction() as conn:
        auth_store.insert_invite(conn, _invite())
        assert auth_store.set_invite_revoked(conn, "INV-1") is True
        assert auth_store.set_invite_revoked(conn, "INV-1") is True
    assert auth_store.get_invite("INV-1")["revoked"] is True
    with auth_store.transaction() as conn:
        assert auth_store.set_invite_revoked(conn, "NOPE") is False


def test_guest_pass_roundtrip(store):
    record = {
        "jti": "j1",
        "label": "difesa",
        "created_at": "2026-01-01T00:00:00+00:00",
        "expires_at": "2026-01-02T00:00:00+00:00",
    }
    with auth_store.transaction() as conn:
        auth_store.insert_guest_pass(conn, record)
    assert auth_store.get_guest_pass("j1")["revoked"] is False
    with auth_store.transaction() as conn:
        assert auth_store.set_guest_pass_revoked(conn, "j1") is True
    assert auth_store.get_guest_pass("j1")["revoked"] is True
    with auth_store.transaction() as conn:
        assert auth_store.set_guest_pass_revoked(conn, "nope") is False


def test_legacy_json_is_imported_once(store):
    path = store / "users.json"
    path.write_text(
        json.dumps(
            {
                "users": [{"username": "vecchio", "password_hash": "h", "role": "admin"}],
                "invites": [{"code": "INV-OLD", "used": True, "role": "user"}],
                "guest_passes": [
                    {"jti": "j1", "label": "", "created_at": "x", "expires_at": "y", "revoked": False}
                ],
            }
        )
    )
    auth_store.initialize(path)
    auth_store.reset_for_tests()
    with auth_store.transaction() as conn:
        auth_store.insert_user(conn, "nuovo", "h", "user")
    auth_store.initialize(path)  # il marker impedisce il secondo import

    assert auth_store.get_user("vecchio")["role"] == "admin"
    assert auth_store.get_invite("INV-OLD")["used"] is True
    assert auth_store.get_invite("INV-OLD")["uses"] == 1
    assert auth_store.get_guest_pass("j1") is not None
    assert auth_store.get_user("nuovo") is not None
    assert path.exists(), "users.json resta come rete di rollback"


def test_legacy_import_does_not_overwrite_existing_rows(store):
    path = store / "users.json"
    path.write_text(json.dumps({"users": [{"username": "mario", "password_hash": "vecchio", "role": "user"}]}))
    with auth_store.transaction() as conn:
        auth_store.insert_user(conn, "mario", "nuovo", "superadmin")
    auth_store.initialize(path)
    assert auth_store.get_user("mario")["password_hash"] == "nuovo"
    assert auth_store.get_user("mario")["role"] == "superadmin"


def test_missing_legacy_json_is_marked(store):
    auth_store.initialize(store / "assente.json")
    auth_store.reset_for_tests()
    with auth_store.transaction() as conn:
        auth_store.insert_user(conn, "mario", "h", "user")
    auth_store.initialize(store / "assente.json")
    assert auth_store.get_user("mario") is not None


def test_malformed_legacy_json_fails_loudly(store):
    path = store / "users.json"
    path.write_text("{ non-json")
    with pytest.raises(json.JSONDecodeError):
        auth_store.initialize(path)
    # nessun marker scritto: un operatore può correggere il file e ripartire
    with auth_store.transaction() as conn:
        assert conn.execute("SELECT 1 FROM auth_meta WHERE key='legacy_json_imported'").fetchone() is None


def test_legacy_json_with_wrong_shape_raises(store):
    path = store / "users.json"
    path.write_text(json.dumps(["non", "un", "dict"]))
    with pytest.raises(ValueError, match="oggetto"):
        auth_store.initialize(path)
