"""Test del layer di persistenza auth (SQLite, transazioni)."""

import json
from datetime import UTC, datetime

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
                "guest_passes": [{"jti": "j1", "label": "", "created_at": "x", "expires_at": "y", "revoked": False}],
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


def test_missing_legacy_json_is_not_marked(store):
    """Un file assente non è una migrazione fatta.

    Marcarlo tale significa che il rimedio ovvio — rimettere `users.json` e
    riavviare — non importa più niente, in silenzio: zero utenti tranne il
    superadmin da env. Il primo avvio su un volume senza JSON deve poter
    riprovare al boot successivo.
    """
    auth_store.initialize(store / "assente.json")
    with auth_store.transaction() as conn:
        row = conn.execute("SELECT value FROM auth_meta WHERE key='legacy_json_imported'").fetchone()
    assert row is None

    # Il file compare (volume rimontato, copia recuperata): il boot dopo lo importa.
    path = store / "assente.json"
    path.write_text(json.dumps({"users": [{"username": "amico", "password_hash": "h", "role": "user"}]}))
    auth_store.reset_for_tests()
    auth_store.initialize(path)
    assert auth_store.get_user("amico") is not None


def test_absent_marker_from_v040_does_not_block_the_import(store):
    """Il marker `absent` scritto dalla v0.4.0 non è una migrazione: si riprova.

    Quel valore marcava la *mancanza* del file, e da lì l'import era morto per
    sempre: rimettere `users.json` e riavviare non importava niente. Ora conta
    solo una data. (In produzione non è successo — l'import del 2026-11-18 c'è
    stato — ma un DB locale o un volume rimontato sì.)
    """
    with auth_store.transaction() as conn:
        auth_store._set_meta(conn, "legacy_json_imported", "absent")
        auth_store.insert_user(conn, "mario", "h", "user")

    path = store / "users.json"
    path.write_text(json.dumps({"users": [{"username": "tardivo", "password_hash": "h", "role": "user"}]}))
    auth_store.reset_for_tests()
    auth_store.initialize(path)
    assert auth_store.get_user("tardivo") is not None
    with auth_store.transaction() as conn:
        row = conn.execute("SELECT value FROM auth_meta WHERE key='legacy_json_imported'").fetchone()
    assert row[0] != "absent", "il marker deve tornare una data dopo un import vero"


def test_imported_marker_stops_a_second_import(store):
    """Con un import già fatto (marker = data), un file nuovo non viene riletto."""
    path = store / "users.json"
    path.write_text(json.dumps({"users": [{"username": "primo", "password_hash": "h", "role": "user"}]}))
    auth_store.initialize(path)

    path.write_text(json.dumps({"users": [{"username": "secondo", "password_hash": "h", "role": "user"}]}))
    auth_store.reset_for_tests()
    auth_store.initialize(path)
    assert auth_store.get_user("primo") is not None
    assert auth_store.get_user("secondo") is None


def test_legacy_invite_ttl_matches_api_policy():
    """`_LEGACY_INVITE_HOURS` è replicato per evitare un import circolare: qui si verifica."""
    from app.core import auth

    assert auth_store._LEGACY_INVITE_HOURS == auth.MAX_INVITE_HOURS


def test_legacy_invites_get_an_expiry_on_import(store):
    """Un codice legacy senza scadenza non deve restare spendibile per sempre."""
    path = store / "users.json"
    path.write_text(
        json.dumps(
            {
                "users": [],
                "invites": [
                    {"code": "INV-LIBERO", "used": False, "role": "user"},
                    {"code": "INV-USATO", "used": True, "role": "user"},
                ],
                "guest_passes": [],
            }
        )
    )
    auth_store.initialize(path)

    invite = auth_store.get_invite("INV-LIBERO")
    assert invite["expires_at"] is not None, "un invito non usato deve scadere"
    assert datetime.fromisoformat(invite["expires_at"]) > datetime.now(UTC)
    # un invito già consumato non ha bisogno di una scadenza: non è più spendibile
    assert auth_store.get_invite("INV-USATO")["expires_at"] is None


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
