"""Test auth core functions."""
import pytest
from unittest.mock import patch

from app.core.auth import verify_password, hash_password, login_user, MAX_LOGIN_ATTEMPTS


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
    def test_lockout_after_max_attempts(self):
        with patch("app.core.auth._load_data") as mock_load:
            mock_load.return_value = {"users": [], "invites": []}
            for i in range(MAX_LOGIN_ATTEMPTS):
                with pytest.raises(Exception, match="Invalid username"):
                    login_user("target_user", "wrong")
            with pytest.raises(Exception, match="locked"):
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
                with pytest.raises(Exception, match="Invalid username"):
                    login_user("valid_user", "wrong")
            # Successful login resets counter
            result = login_user("valid_user", "correct")
            assert result["username"] == "valid_user"
            # Now should be able to fail again
            with pytest.raises(Exception, match="Invalid username"):
                login_user("valid_user", "wrong")
