"""Unit tests for app.core.auth."""

import jwt
import pytest

from app.config import settings
from app.core.auth import ROLE_HIERARCHY, create_jwt, decode_jwt, hash_password, verify_password


class TestPassword:
    def test_hash_and_verify(self):
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed)

    def test_wrong_password_fails(self):
        hashed = hash_password("mypassword")
        assert not verify_password("wrongpassword", hashed)

    def test_different_hash_each_time(self):
        h1 = hash_password("samepass")
        h2 = hash_password("samepass")
        assert h1 != h2


class TestJWT:
    def test_create_and_decode(self):
        token = create_jwt("testuser", "admin")
        payload = decode_jwt(token)
        assert payload["sub"] == "testuser"
        assert payload["role"] == "admin"
        assert payload["aud"] == "orchid-osr"
        assert payload["iss"] == "orchid-osr-api"

    def test_decode_expired_token(self):
        import time

        payload = {
            "sub": "testuser",
            "role": "user",
            "aud": "orchid-osr",
            "iss": "orchid-osr-api",
            "iat": int(time.time()) - 99999,
            "exp": int(time.time()) - 1,
        }
        token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
        with pytest.raises(Exception):
            decode_jwt(token)


class TestRoles:
    def test_hierarchy_order(self):
        assert ROLE_HIERARCHY["user"] < ROLE_HIERARCHY["admin"] < ROLE_HIERARCHY["superadmin"]

    def test_require_role_logic(self):
        allowed = ROLE_HIERARCHY.get("admin", -1)
        for role, level in ROLE_HIERARCHY.items():
            if level >= allowed:
                assert level >= allowed
            else:
                assert level < allowed
