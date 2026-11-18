"""Test auth core functions (persistenza SQLite reale, DB temporaneo)."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

import app.core.auth as auth_mod
from app.config import settings
from app.core import auth_store
from app.core.auth import MAX_LOGIN_ATTEMPTS, hash_password, login_user, verify_password


@pytest.fixture(autouse=True)
def _reset_login_cache():
    auth_mod.LOGIN_ATTEMPTS_CACHE.clear()
    auth_mod._CACHE_LAST_SYNC = 0.0
    yield


def _make_user(username: str, password: str = "Password1", role: str = "user") -> None:
    with auth_store.transaction() as conn:
        auth_store.insert_user(conn, username, hash_password(password), role)


def _make_invite(code: str, **overrides) -> None:
    record = {
        "code": code,
        "role": "user",
        "used": False,
        "used_by": None,
        "created_at": datetime.now(UTC).isoformat(),
        "expires_at": None,
        "max_uses": 1,
        "uses": 0,
        "label": "",
        "revoked": False,
    }
    record.update(overrides)
    with auth_store.transaction() as conn:
        auth_store.insert_invite(conn, record)


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = hash_password("test1234")
        assert verify_password("test1234", hashed)
        assert not verify_password("wrong", hashed)

    def test_different_salts(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2


class TestLoginLockout:
    def test_lockout_after_max_attempts(self, auth_store_db):
        for _ in range(MAX_LOGIN_ATTEMPTS):
            with pytest.raises(HTTPException, match="Credenziali non valide"):
                login_user("target_user", "wrong")
        with pytest.raises(HTTPException, match="locked"):
            login_user("target_user", "wrong")

    def test_successful_login_resets_counter(self, auth_store_db):
        _make_user("valid_user", "correct")
        for _ in range(MAX_LOGIN_ATTEMPTS - 1):
            with pytest.raises(HTTPException, match="Credenziali non valide"):
                login_user("valid_user", "wrong")
        result = login_user("valid_user", "correct")
        assert result["username"] == "valid_user"
        with pytest.raises(HTTPException, match="Credenziali non valide"):
            login_user("valid_user", "wrong")

    def test_login_reads_hash_from_imported_json(self, auth_store_db):
        """L'hash bcrypt scritto in users.json resta valido dopo l'import."""
        import json

        from app.core import auth as auth_module

        hashed = hash_password("Password1")
        (auth_store_db.parent / "users.json").write_text(
            json.dumps({"users": [{"username": "mario", "password_hash": hashed, "role": "user"}]})
        )
        auth_module.USERS_PATH = auth_store_db.parent / "users.json"
        auth_store.reset_for_tests()
        assert login_user("mario", "Password1")["role"] == "user"


class TestTimedInvites:
    def test_multiuse_until_exhausted(self, auth_store_db):
        code = auth_mod.generate_invite(role="user", count=1, max_uses=3, label="difesa")[0]
        for n in range(3):
            assert auth_mod.register_user(code, f"u{n}", "Password1")["role"] == "user"
        with pytest.raises(HTTPException, match="already used"):
            auth_mod.register_user(code, "u3", "Password1")

    def test_expired_rejected(self, auth_store_db):
        _make_invite(
            "OLD-1",
            expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            max_uses=5,
        )
        with pytest.raises(HTTPException, match="expired"):
            auth_mod.register_user("OLD-1", "late", "Password1")

    def test_malformed_expiry_is_rejected(self, auth_store_db):
        _make_invite("BAD-1", expires_at="not-a-date")
        with pytest.raises(HTTPException, match="invalid"):
            auth_mod.register_user("BAD-1", "late", "Password1")

    def test_revoked_rejected_and_revoke_endpoint(self, auth_store_db):
        code = auth_mod.generate_invite(role="user", count=1, max_uses=10)[0]
        auth_mod.revoke_invite(code)
        with pytest.raises(HTTPException, match="revoked"):
            auth_mod.register_user(code, "nope", "Password1")
        with pytest.raises(HTTPException, match="not found"):
            auth_mod.revoke_invite("MISSING-CODE")

    def test_legacy_invite_still_single_use(self, auth_store_db):
        _make_invite("LEG-1")
        auth_mod.register_user("LEG-1", "first", "Password1")
        with pytest.raises(HTTPException, match="already used"):
            auth_mod.register_user("LEG-1", "second", "Password1")

    def test_custom_code_duplicate_rejected(self, auth_store_db):
        auth_mod.generate_invite(count=1, code="DIFESA-1")
        with pytest.raises(HTTPException, match="already exists"):
            auth_mod.generate_invite(count=1, code="DIFESA-1")

    def test_invite_ttl_is_capped(self, auth_store_db):
        with pytest.raises(HTTPException, match="expires_in_hours"):
            auth_mod.generate_invite(expires_in_hours=auth_mod.MAX_INVITE_HOURS + 1)

    def test_generated_invite_has_expiry(self, auth_store_db):
        code = auth_mod.generate_invite(expires_in_hours=24)[0]
        invite = auth_store.get_invite(code)
        assert invite["expires_at"] is not None


class TestGuestPass:
    def test_create_redeem_revoke(self, auth_store_db):
        rec = auth_mod.create_guest_pass(expires_in_hours=24, label="difesa")
        assert rec["pass"] and rec["expires_at"] and not rec["revoked"]
        assert any(u["username"] == "ospite" and u["role"] == "user" for u in auth_store.list_users())
        creds = auth_mod.redeem_guest_pass(rec["pass"])
        assert creds == {"username": "ospite", "role": "user"}
        auth_mod.revoke_guest_pass(rec["jti"])
        with pytest.raises(HTTPException, match="revoked"):
            auth_mod.redeem_guest_pass(rec["pass"])
        with pytest.raises(HTTPException, match="not found"):
            auth_mod.revoke_guest_pass("nope")

    def test_tampered_or_wrong_purpose_rejected(self, auth_store_db):
        rec = auth_mod.create_guest_pass(expires_in_hours=24)
        with pytest.raises(HTTPException, match="Invalid"):
            auth_mod.redeem_guest_pass(rec["pass"] + "x")
        with pytest.raises(HTTPException, match="Invalid"):
            auth_mod.redeem_guest_pass(auth_mod.create_jwt("ospite", "user"))

    def test_expired_pass_rejected(self, auth_store_db):
        import time

        import jwt as _jwt

        now = int(time.time())
        payload = {
            "sub": "ospite",
            "role": "user",
            "purpose": "guest-pass",
            "jti": "vecchio",
            "aud": "orchid-osr",
            "iss": "orchid-osr-api",
            "iat": now - 7200,
            "exp": now - 3600,
        }
        token = _jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
        with auth_store.transaction() as conn:
            auth_store.insert_guest_pass(
                conn,
                {
                    "jti": "vecchio",
                    "label": "",
                    "created_at": datetime.now(UTC).isoformat(),
                    "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                },
            )
        with pytest.raises(HTTPException, match="Invalid or expired"):
            auth_mod.redeem_guest_pass(token)

    def test_record_expiry_is_checked_even_with_valid_jwt(self, auth_store_db):
        """Il JWT vale 24h, ma il record può essere stato creato con meno: vince il record."""
        rec = auth_mod.create_guest_pass(expires_in_hours=24)
        with auth_store.transaction() as conn:
            conn.execute(
                "UPDATE guest_passes SET expires_at = ? WHERE jti = ?",
                ((datetime.now(UTC) - timedelta(minutes=1)).isoformat(), rec["jti"]),
            )
        with pytest.raises(HTTPException, match="Pass expired"):
            auth_mod.redeem_guest_pass(rec["pass"])

    def test_guest_pass_max_hours(self, auth_store_db):
        with pytest.raises(HTTPException, match="expires_in_hours"):
            auth_mod.create_guest_pass(expires_in_hours=auth_mod.MAX_GUEST_PASS_HOURS + 1)
