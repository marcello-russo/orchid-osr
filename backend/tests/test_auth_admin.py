"""Test superadmin bootstrap, user deletion, and role hierarchy (SQLite reale)."""

import pytest
from fastapi import HTTPException

from app.core import auth_store
from app.core.auth import delete_user, ensure_superadmin_exists

# ── helpers ────────────────────────────────────────────────


def _make_user(username: str, role: str = "user") -> None:
    with auth_store.transaction() as conn:
        auth_store.insert_user(conn, username, "fakehash", role)


def _make_invite(code: str = "INV-TEST", *, used_by: str | None = None) -> None:
    with auth_store.transaction() as conn:
        auth_store.insert_invite(
            conn,
            {
                "code": code,
                "role": "user",
                "used": used_by is not None,
                "used_by": used_by,
                "created_at": "2026-01-01T00:00:00+00:00",
                "expires_at": None,
                "max_uses": 1,
                "uses": 1 if used_by else 0,
                "label": "",
                "revoked": False,
            },
        )


@pytest.fixture
def seeded(auth_store_db):
    _make_user("super", "superadmin")
    _make_user("alfa", "admin")
    _make_user("beta", "user")
    _make_invite("INV-UNUSED")
    _make_invite("INV-USED", used_by="beta")
    return auth_store_db


# ── delete_user ────────────────────────────────────────────


class TestDeleteUser:
    def test_superadmin_deletes_user(self, seeded):
        delete_user("beta", "super")
        assert auth_store.get_user("beta") is None

    def test_superadmin_deletes_admin(self, seeded):
        delete_user("alfa", "super")
        assert auth_store.get_user("alfa") is None

    def test_admin_deletes_user(self, seeded):
        delete_user("beta", "alfa")
        assert auth_store.get_user("beta") is None

    def test_admin_cannot_delete_admin(self, seeded):
        _make_user("gamma", "admin")
        with pytest.raises(HTTPException, match="Cannot delete a user with equal or higher role"):
            delete_user("gamma", "alfa")
        assert auth_store.get_user("gamma") is not None

    def test_admin_cannot_delete_superadmin(self, seeded):
        with pytest.raises(HTTPException, match="Cannot delete a user with equal or higher role"):
            delete_user("super", "alfa")
        assert auth_store.get_user("super") is not None

    def test_cannot_delete_self(self, seeded):
        with pytest.raises(HTTPException, match="Cannot delete yourself"):
            delete_user("super", "super")
        assert auth_store.get_user("super") is not None

    def test_delete_nonexistent_user(self, seeded):
        with pytest.raises(HTTPException, match="User not found"):
            delete_user("nonexistent", "super")

    def test_delete_resets_invites(self, seeded):
        """L'invito torna utilizzabile: *sia* `used` che `uses` vanno azzerati."""
        delete_user("beta", "super")
        invite = auth_store.get_invite("INV-USED")
        assert invite is not None
        assert invite["used"] is False
        assert invite["uses"] == 0
        assert invite["used_by"] is None


# ── ensure_superadmin_exists ───────────────────────────────


class TestEnsureSuperadminExists:
    def _patch_settings(self, monkeypatch, username: str = "super", password: str = "pass"):
        monkeypatch.setattr("app.core.auth.settings.superadmin_username", username)
        monkeypatch.setattr("app.core.auth.settings.superadmin_password", password)

    def test_creates_superadmin_when_missing(self, auth_store_db, monkeypatch):
        self._patch_settings(monkeypatch)
        ensure_superadmin_exists()
        user = auth_store.get_user("super")
        assert user is not None
        assert user["role"] == "superadmin"

    def test_updates_existing_user_to_superadmin(self, auth_store_db, monkeypatch):
        _make_user("super", "admin")
        self._patch_settings(monkeypatch)
        ensure_superadmin_exists()
        assert auth_store.get_user("super")["role"] == "superadmin"

    def test_skips_when_credentials_missing(self, auth_store_db, monkeypatch):
        self._patch_settings(monkeypatch, username="", password="")
        ensure_superadmin_exists()
        assert auth_store.list_users() == []

    def test_preserves_other_users(self, auth_store_db, monkeypatch):
        _make_user("alfa")
        _make_user("beta")
        self._patch_settings(monkeypatch)
        ensure_superadmin_exists()
        assert {u["username"] for u in auth_store.list_users()} == {"alfa", "beta", "super"}
