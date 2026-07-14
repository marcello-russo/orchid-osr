"""Tests for ensure_admin_exists in auth.py."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_users(users=None, invites=None):
    """Create a temporary users.json with given data."""
    data = {"users": users or [], "invites": invites or []}
    f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
    json.dump(data, f)
    f.close()
    return f.name


def test_creates_admin_when_missing():
    """When no admin exists, ensure_admin_exists should create one."""
    path = _make_users(users=[], invites=[])
    with patch("app.core.auth.USERS_PATH", Path(path)):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.admin_username = "admin"
            mock_settings.admin_password = "secret123"
            mock_settings.hf_configured = False

            from app.core.auth import ensure_admin_exists

            ensure_admin_exists()

            with open(path) as f:
                data = json.load(f)
            admins = [u for u in data["users"] if u["role"] == "admin"]
            assert len(admins) == 1
            assert admins[0]["username"] == "admin"


def test_updates_password_when_exists():
    """When admin user already exists, ensure_admin_exists should update password."""
    path = _make_users(
        users=[{"username": "admin", "password_hash": "old_hash", "role": "admin"}],
        invites=[],
    )
    with patch("app.core.auth.USERS_PATH", Path(path)):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.admin_username = "admin"
            mock_settings.admin_password = "new_secret"
            mock_settings.hf_configured = False

            from app.core.auth import ensure_admin_exists, verify_password

            ensure_admin_exists()

            with open(path) as f:
                data = json.load(f)
            admin = next(u for u in data["users"] if u["username"] == "admin")
            assert verify_password("new_secret", admin["password_hash"]) is True


def test_skips_when_no_env():
    """When ADMIN_USERNAME/PASSWORD not set, should skip without error."""
    path = _make_users(users=[], invites=[])
    with patch("app.core.auth.USERS_PATH", Path(path)):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.admin_username = ""
            mock_settings.admin_password = ""
            mock_settings.hf_configured = False

            from app.core.auth import ensure_admin_exists

            ensure_admin_exists()

            with open(path) as f:
                data = json.load(f)
            assert len(data["users"]) == 0


def test_preserves_existing_users():
    """Ensure_admin_exists should not delete other users."""
    path = _make_users(
        users=[{"username": "marcello", "password_hash": "hash", "role": "admin"}],
        invites=[],
    )
    with patch("app.core.auth.USERS_PATH", Path(path)):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.admin_username = "newadmin"
            mock_settings.admin_password = "pwd"
            mock_settings.hf_configured = False

            from app.core.auth import ensure_admin_exists

            ensure_admin_exists()

            with open(path) as f:
                data = json.load(f)
            usernames = [u["username"] for u in data["users"]]
            assert "marcello" in usernames
            assert "newadmin" in usernames
