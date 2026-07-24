import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import settings


@pytest.fixture(autouse=True, scope="session")
def _no_rate_limits():
    """Disabilita slowapi rate limiting e login-attempt tracking per tutto il test session."""
    from app.api.routes.auth import limiter

    limiter.enabled = False

    import app.core.auth as auth_mod

    # login_attempts tracking usa DB SQLite condiviso → va disabilitato nei test
    auth_mod._load_attempts = lambda: {}
    auth_mod._save_attempts = lambda _u, _c: None
    auth_mod.LOGIN_ATTEMPTS_CACHE.clear()
    yield


@pytest.fixture
def client():
    """TestClient senza autenticazione."""
    return TestClient(app)


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    """Crea database auth temporaneo con superadmin e invite codes."""
    from app.core import auth as auth_module

    db_path = tmp_path / "auth_db.json"
    db_path.write_text(json.dumps({"invites": [], "users": []}))
    monkeypatch.setattr(auth_module, "USERS_PATH", db_path)

    monkeypatch.setattr(settings, "superadmin_username", "testsuperadmin")
    monkeypatch.setattr(settings, "superadmin_password", "TestAdmin2026!")
    auth_module.ensure_superadmin_exists()

    auth_module.generate_invite(role="user", count=5)
    auth_module.generate_invite(role="admin", count=2)

    return db_path


@pytest.fixture
def superadmin_token(auth_db, client):
    """Token di autenticazione come superadmin."""
    resp = client.post(
        "/auth/login",
        json={
            "username": "testsuperadmin",
            "password": "TestAdmin2026!",
        },
    )
    return resp.json()["token"]


@pytest.fixture
def user_token(auth_db, client):
    """Registra e logga un utente normale, restituisce il token."""
    auth_data = json.loads(auth_db.read_text())
    invite = next(i for i in auth_data["invites"] if not i["used"] and i["role"] == "user")

    client.post(
        "/auth/register",
        json={
            "username": "testuser",
            "password": "Password1",
            "invite_code": invite["code"],
        },
    )
    resp = client.post(
        "/auth/login",
        json={
            "username": "testuser",
            "password": "Password1",
        },
    )
    return resp.json()["token"]
