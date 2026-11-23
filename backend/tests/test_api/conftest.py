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

    # login_attempts tracking usa DB SQLite: nei test API il lockout è coperto
    # dai test unitari, qui non deve scrivere su un DB condiviso.
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
    """DB auth SQLite temporaneo con superadmin e inviti (nessun users.json)."""
    from app.core import auth as auth_module
    from app.core import auth_store
    from app.core import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "orchid.db"))
    monkeypatch.setattr(db_module, "_SCHEMA_READY", set())
    monkeypatch.setattr(auth_module, "USERS_PATH", tmp_path / "users.json")
    auth_store.reset_for_tests()

    monkeypatch.setattr(settings, "superadmin_username", "testsuperadmin")
    monkeypatch.setattr(settings, "superadmin_password", "TestAdmin2026!")
    auth_module.ensure_superadmin_exists()

    auth_module.generate_invite(role="user", count=5)
    auth_module.generate_invite(role="admin", count=2)

    yield tmp_path / "orchid.db"
    auth_store.reset_for_tests()


@pytest.fixture
def next_invite():
    """Codice di un invito non ancora usato del ruolo richiesto."""

    def _next(role: str = "user") -> str:
        from app.core import auth_store

        invite = next(i for i in auth_store.list_invites() if not i["used"] and i["role"] == role)
        return str(invite["code"])

    return _next


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
def user_token(auth_db, client, next_invite):
    """Registra e logga un utente normale, restituisce il token."""
    client.post(
        "/auth/register",
        json={
            "username": "testuser",
            "password": "Password1",
            "invite_code": next_invite(),
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
