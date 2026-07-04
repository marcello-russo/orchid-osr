"""Test auth core functions."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.core.auth import MAX_LOGIN_ATTEMPTS, hash_password, login_user, verify_password


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
        with patch("app.core.auth._load_data") as mock_load:
            mock_load.return_value = {"users": [], "invites": []}
            for _ in range(MAX_LOGIN_ATTEMPTS):
                with pytest.raises(HTTPException, match="Credenziali non valide"):
                    login_user("target_user", "wrong")
            with pytest.raises(HTTPException, match="locked"):
                login_user("target_user", "wrong")

    def test_successful_login_resets_counter(self):
        with patch("app.core.auth._load_data") as mock_load:
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
