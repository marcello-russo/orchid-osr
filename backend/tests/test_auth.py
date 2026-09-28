"""Test auth core functions."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.core.auth import LOGIN_ATTEMPTS_CACHE, MAX_LOGIN_ATTEMPTS, hash_password, login_user, verify_password


@pytest.fixture(autouse=True)
def _reset_login_cache():
    """Pulisce la cache dei tentativi di login prima di ogni test."""
    import app.core.auth as auth_mod

    LOGIN_ATTEMPTS_CACHE.clear()
    auth_mod._CACHE_LAST_SYNC = 0.0
    yield


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = hash_password("test1234")
        assert verify_password("test1234", hashed)
        assert not verify_password("wrong", hashed)

    def test_different_salts(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # bcrypt salts should differ


class TestLoginLockout:
    def _assert_auth_error(self, match: str) -> None:
        return pytest.raises(HTTPException, match=match)

    def test_lockout_after_max_attempts(self):
        with (
            patch("app.core.auth._load_data") as mock_load,
            patch("app.core.auth._load_attempts", return_value={}),
            patch("app.core.auth._save_attempts"),
        ):
            mock_load.return_value = {"users": [], "invites": []}
            for _ in range(MAX_LOGIN_ATTEMPTS):
                with pytest.raises(HTTPException, match="Credenziali non valide"):
                    login_user("target_user", "wrong")
            with pytest.raises(HTTPException, match="locked"):
                login_user("target_user", "wrong")

    def test_successful_login_resets_counter(self):
        with (
            patch("app.core.auth._load_data") as mock_load,
            patch("app.core.auth._load_attempts", return_value={}),
            patch("app.core.auth._save_attempts"),
        ):
            mock_user = {
                "username": "valid_user",
                "password_hash": hash_password("correct"),
                "role": "user",
            }
            mock_load.return_value = {"users": [mock_user], "invites": []}
            # Failed attempts
            for _ in range(MAX_LOGIN_ATTEMPTS - 1):
                with pytest.raises(HTTPException, match="Credenziali non valide"):
                    login_user("valid_user", "wrong")
            # Successful login resets counter
            result = login_user("valid_user", "correct")
            assert result["username"] == "valid_user"
            # Now should be able to fail again
            with pytest.raises(HTTPException, match="Credenziali non valide"):
                login_user("valid_user", "wrong")


class TestTimedInvites:
    """Inviti multiuso con scadenza e revoca (defense pass)."""

    def _store(self, invites):
        store = {"users": [], "invites": invites}
        return store

    def _patched(self, store):
        import app.core.auth as auth_mod

        return (
            patch.object(auth_mod, "_load_data", return_value=store),
            patch.object(auth_mod, "_save_data"),
        )

    def test_multiuse_until_exhausted(self):
        import app.core.auth as auth_mod

        store = self._store([])
        pl, ps = self._patched(store)
        with pl, ps:
            codes = auth_mod.generate_invite(role="user", count=1, max_uses=3, label="difesa")
            code = codes[0]
            for n in range(3):
                r = auth_mod.register_user(code, f"u{n}", "Password1")
                assert r["role"] == "user"
            with pytest.raises(HTTPException, match="already used"):
                auth_mod.register_user(code, "u3", "Password1")

    def test_expired_rejected(self):
        from datetime import UTC, datetime, timedelta

        import app.core.auth as auth_mod

        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        store = self._store(
            [
                {
                    "code": "OLD-1",
                    "used": False,
                    "role": "user",
                    "expires_at": past,
                    "max_uses": 5,
                    "uses": 0,
                    "label": "",
                    "revoked": False,
                }
            ]
        )
        pl, ps = self._patched(store)
        with pl, ps:
            with pytest.raises(HTTPException, match="expired"):
                auth_mod.register_user("OLD-1", "late", "Password1")

    def test_revoked_rejected_and_revoke_endpoint(self):
        import app.core.auth as auth_mod

        store = self._store([])
        pl, ps = self._patched(store)
        with pl, ps:
            code = auth_mod.generate_invite(role="user", count=1, max_uses=10)[0]
            auth_mod.revoke_invite(code)
            with pytest.raises(HTTPException, match="revoked"):
                auth_mod.register_user(code, "nope", "Password1")
            with pytest.raises(HTTPException, match="not found"):
                auth_mod.revoke_invite("MISSING-CODE")

    def test_legacy_invite_still_single_use(self):
        import app.core.auth as auth_mod

        store = self._store([{"code": "LEG-1", "used": False, "role": "user"}])
        pl, ps = self._patched(store)
        with pl, ps:
            auth_mod.register_user("LEG-1", "first", "Password1")
            with pytest.raises(HTTPException, match="already used"):
                auth_mod.register_user("LEG-1", "second", "Password1")

    def test_custom_code_duplicate_rejected(self):
        import app.core.auth as auth_mod

        store = self._store([])
        pl, ps = self._patched(store)
        with pl, ps:
            auth_mod.generate_invite(count=1, code="DIFESA-1")
            with pytest.raises(HTTPException, match="already exists"):
                auth_mod.generate_invite(count=1, code="DIFESA-1")
