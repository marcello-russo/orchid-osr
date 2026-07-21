"""Test superadmin bootstrap, user deletion, and role hierarchy."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.core.auth import delete_user, ensure_superadmin_exists

# ── helpers ────────────────────────────────────────────────


def _make_user(username: str, role: str = "user") -> dict:
    return {"username": username, "password_hash": "fakehash", "role": role}


def _make_invite(code: str = "INV-TEST", used: bool = False, used_by: str = "") -> dict:
    inv = {"code": code, "used": used, "role": "user", "created_at": "2026-01-01T00:00:00"}
    if used and used_by:
        inv["used"] = True
        inv["used_by"] = used_by
    return inv


def _default_data() -> dict:
    return {
        "users": [
            _make_user("super", "superadmin"),
            _make_user("alfa", "admin"),
            _make_user("beta", "user"),
        ],
        "invites": [
            _make_invite("INV-UNUSED"),
            _make_invite("INV-USED", used=True, used_by="beta"),
        ],
    }


# ── delete_user ────────────────────────────────────────────


class TestDeleteUser:
    def test_superadmin_deletes_user(self):
        """Superadmin può eliminare un user."""
        data = _default_data()
        with patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)):
            delete_user("beta", "super")
        users = [u["username"] for u in data["users"]]
        assert "beta" not in users

    def test_superadmin_deletes_admin(self):
        """Superadmin può eliminare un admin."""
        data = _default_data()
        with patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)):
            delete_user("alfa", "super")
        users = [u["username"] for u in data["users"]]
        assert "alfa" not in users

    def test_admin_deletes_user(self):
        """Admin può eliminare un user."""
        data = _default_data()
        with patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)):
            delete_user("beta", "alfa")
        users = [u["username"] for u in data["users"]]
        assert "beta" not in users

    def test_admin_cannot_delete_admin(self):
        """Admin NON può eliminare un altro admin."""
        data = _default_data()
        data["users"].append(_make_user("gamma", "admin"))
        with patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)):
            with pytest.raises(HTTPException, match="Cannot delete a user with equal or higher role"):
                delete_user("gamma", "alfa")

    def test_admin_cannot_delete_superadmin(self):
        """Admin NON può eliminare un superadmin."""
        data = _default_data()
        with patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)):
            with pytest.raises(HTTPException, match="Cannot delete a user with equal or higher role"):
                delete_user("super", "alfa")

    def test_cannot_delete_self(self):
        """Un utente non può eliminare sé stesso."""
        data = _default_data()
        with patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)):
            with pytest.raises(HTTPException, match="Cannot delete yourself"):
                delete_user("super", "super")

    def test_delete_nonexistent_user(self):
        """Eliminare un utente inesistente → 404."""
        data = _default_data()
        with patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)):
            with pytest.raises(HTTPException, match="User not found"):
                delete_user("nonexistent", "super")

    def test_delete_resets_invites(self):
        """Gli inviti usati dall'utente eliminato vengono resettati."""
        data = _default_data()
        with patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)):
            delete_user("beta", "super")
        invites = [i for i in data["invites"] if i["code"] == "INV-USED"]
        assert len(invites) == 1
        assert invites[0]["used"] is False
        assert "used_by" not in invites[0]


# ── ensure_superadmin_exists ───────────────────────────────


class TestEnsureSuperadminExists:
    def _patch_settings(self, username: str = "super", password: str = "pass"):
        return patch(
            "app.core.auth.settings",
            superadmin_username=username,
            superadmin_password=password,
        )

    def test_creates_superadmin_when_missing(self):
        """Se il superadmin non esiste, viene creato."""
        data = {"users": [_make_user("user1")], "invites": []}
        with (
            patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)),
            self._patch_settings(),
        ):
            ensure_superadmin_exists()
        users = {u["username"]: u["role"] for u in data["users"]}
        assert "super" in users
        assert users["super"] == "superadmin"

    def test_updates_existing_user_to_superadmin(self):
        """Se l'utente esiste già, il ruolo viene aggiornato a superadmin."""
        data = {"users": [_make_user("super", "admin")], "invites": []}
        with (
            patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)),
            self._patch_settings(),
        ):
            ensure_superadmin_exists()
        user = next(u for u in data["users"] if u["username"] == "super")
        assert user["role"] == "superadmin"

    def test_skips_when_credentials_missing(self):
        """Se SUPERADMIN_USERNAME/PASSWORD non sono configurati, non fa nulla."""
        data = {"users": [], "invites": []}
        with (
            patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)),
            self._patch_settings(username="", password=""),
        ):
            ensure_superadmin_exists()
        assert len(data["users"]) == 0

    def test_preserves_other_users(self):
        """La funzione non elimina altri utenti."""
        data = {"users": [_make_user("alfa"), _make_user("beta")], "invites": []}
        with (
            patch("app.core.auth._atomic_modify", side_effect=lambda fn: fn(data)),
            self._patch_settings(),
        ):
            ensure_superadmin_exists()
        assert len(data["users"]) == 3
