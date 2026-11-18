"""Comandi CLI auth: stesse funzioni dell'API, DB temporaneo."""

from types import SimpleNamespace

import pytest

from app.cli.auth_commands import handle_auth_commands
from app.core import auth_store


@pytest.fixture(autouse=True)
def _no_superadmin(monkeypatch):
    """Il bootstrap superadmin non deve toccare il DB temporaneo del test."""
    monkeypatch.setattr("app.core.auth.settings.superadmin_username", "")
    monkeypatch.setattr("app.core.auth.settings.superadmin_password", "")


def _args(**overrides):
    base = {
        "list_users": False,
        "gen_invites": None,
        "invite_hours": 72,
        "add_user": None,
        "username": None,
        "password": None,
        "role": "user",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_returns_none_when_not_an_auth_command(auth_store_db):
    assert handle_auth_commands(_args()) is None


def test_add_user_positional(auth_store_db, capsys):
    assert handle_auth_commands(_args(add_user="cli-user", password="Password1")) == 0
    assert auth_store.get_user("cli-user")["role"] == "user"
    assert "created" in capsys.readouterr().out


def test_add_user_via_username_flag(auth_store_db):
    args = _args(username="flag-user", password="Password1", role="admin")
    assert handle_auth_commands(args) == 0
    assert auth_store.get_user("flag-user")["role"] == "admin"


def test_add_user_without_password_fails(auth_store_db, capsys):
    assert handle_auth_commands(_args(add_user="x")) == 1
    assert "--password is required" in capsys.readouterr().out
    assert auth_store.get_user("x") is None


def test_add_duplicate_user_fails(auth_store_db, capsys):
    assert handle_auth_commands(_args(add_user="dup", password="Password1")) == 0
    assert handle_auth_commands(_args(add_user="dup", password="Password1")) == 1
    assert "already exists" in capsys.readouterr().out


def test_gen_invites_has_default_expiry(auth_store_db, capsys):
    assert handle_auth_commands(_args(gen_invites=2)) == 0
    codes = capsys.readouterr().out.split()
    assert len(codes) == 2
    for code in codes:
        invite = auth_store.get_invite(code)
        assert invite is not None
        assert invite["expires_at"] is not None
        assert invite["role"] == "user"


def test_gen_invites_zero_means_no_expiry(auth_store_db, capsys):
    assert handle_auth_commands(_args(gen_invites=1, invite_hours=0)) == 0
    code = capsys.readouterr().out.strip()
    assert auth_store.get_invite(code)["expires_at"] is None


def test_gen_invites_over_cap_is_rejected(auth_store_db, capsys):
    assert handle_auth_commands(_args(gen_invites=1, invite_hours=1000)) == 1
    assert "invite-hours" in capsys.readouterr().out


def test_list_users_prints_roles_and_status(auth_store_db, capsys):
    with auth_store.transaction() as conn:
        auth_store.insert_user(conn, "mario", "hash", "admin")
    assert handle_auth_commands(_args(gen_invites=1)) == 0
    code = capsys.readouterr().out.strip()
    assert handle_auth_commands(_args(list_users=True)) == 0
    out = capsys.readouterr().out
    assert "mario (admin)" in out
    assert f"{code} [available]" in out
