import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
import jwt
from fastapi import HTTPException, status

from app.config import settings

USERS_PATH = Path(os.environ.get("USERS_PATH", "data/users.json"))
_HF_DATASET_REPO = "marcellorusso/orchid-ncd-dataset"
JWT_ALGORITHM = "HS256"
_LOGIN_ATTEMPTS: OrderedDict[str, int] = OrderedDict()
LOGIN_ATTEMPTS_CACHE: dict[str, int] = {}
_CACHE_LAST_SYNC: float = 0.0
MAX_LOGIN_ATTEMPTS = 5
_MAX_TRACKED_USERS = 1000
_LOGIN_LOCK = threading.Lock()


def _load_attempts() -> dict[str, int]:
    global _CACHE_LAST_SYNC
    if time.time() - _CACHE_LAST_SYNC < 5.0:
        return LOGIN_ATTEMPTS_CACHE
    try:
        from app.core.db import get_db

        conn = get_db()
        rows = conn.execute("SELECT username, attempts FROM login_attempts").fetchall()
        LOGIN_ATTEMPTS_CACHE.clear()
        LOGIN_ATTEMPTS_CACHE.update({r[0]: r[1] for r in rows})
        _CACHE_LAST_SYNC = time.time()
    except Exception:
        pass
    return LOGIN_ATTEMPTS_CACHE


def _save_attempts(username: str, count: int) -> None:
    try:
        from app.core.db import get_db

        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO login_attempts (username, attempts, last_attempt) VALUES (?, ?, ?)",
            (username, count, time.time()),
        )
        conn.commit()
    except Exception:
        pass


_DATA_LOCK = threading.Lock()


def _sync_to_hf() -> None:
    """Upload users.json to private HF dataset repo as backup."""
    if not settings.hf_configured:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=settings.hf_token)
        api.upload_file(
            path_or_fileobj=str(USERS_PATH),
            path_in_repo="users.json",
            repo_id=_HF_DATASET_REPO,
            repo_type="dataset",
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"HF sync failed: {e}")


def _load_data() -> dict:
    if USERS_PATH.exists():
        return json.loads(USERS_PATH.read_text())
    return {"invites": [], "users": []}


def _save_data(data: dict) -> None:
    USERS_PATH.write_text(json.dumps(data, indent=2, default=str))
    _sync_to_hf()


def _atomic_modify(modifier):
    """Read-modify-write atomico per users.json."""
    with _DATA_LOCK:
        data = _load_data()
        modifier(data)
        _save_data(data)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_jwt(username: str, role: str) -> str:
    import time

    payload = {
        "sub": username,
        "role": role,
        "aud": "orchid-ncd",
        "iss": "orchid-ncd-api",
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[JWT_ALGORITHM],
            audience="orchid-ncd",
            issuer="orchid-ncd-api",
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


def register_user(invite_code: str, username: str, password: str) -> dict:
    result = {"username": username, "role": "user"}

    def _modify(data):
        invite = next((i for i in data["invites"] if i["code"] == invite_code), None)
        if not invite:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid invite code")
        if invite["used"]:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invite code already used")
        if any(u["username"] == username for u in data["users"]):
            raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")

        invite["used"] = True
        invite["used_by"] = username
        role = invite["role"]
        data["users"].append(
            {
                "username": username,
                "password_hash": hash_password(password),
                "role": role,
            }
        )
        result["role"] = role

    _atomic_modify(_modify)
    return result


def login_user(username: str, password: str) -> dict:
    _load_attempts()
    with _LOGIN_LOCK:
        if LOGIN_ATTEMPTS_CACHE.get(username, 0) >= MAX_LOGIN_ATTEMPTS:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Account temporarily locked. Try again later.")

    with _DATA_LOCK:
        data = _load_data()
    user = next((u for u in data["users"] if u["username"] == username), None)
    if not user or not verify_password(password, user["password_hash"]):
        with _LOGIN_LOCK:
            count = LOGIN_ATTEMPTS_CACHE.get(username, 0) + 1
            LOGIN_ATTEMPTS_CACHE[username] = count
        _save_attempts(username, count)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide. Username o password errati.",
        )

    with _LOGIN_LOCK:
        LOGIN_ATTEMPTS_CACHE.pop(username, None)
    _save_attempts(username, 0)
    return {"username": user["username"], "role": user["role"]}


def ensure_admin_exists() -> None:
    if not settings.admin_username or not settings.admin_password:
        logger = logging.getLogger(__name__)
        logger.warning(
            "ADMIN_USERNAME/ADMIN_PASSWORD not configured — admin account disabled. "
            "Set these via .env, HF Spaces Secrets, or ensure users.json provides an admin."
        )
        return

    def _modify(data):
        existing = next((u for u in data["users"] if u["username"] == settings.admin_username), None)
        if existing:
            existing["password_hash"] = hash_password(settings.admin_password)
            existing["role"] = "admin"
            return
        data["users"].append(
            {
                "username": settings.admin_username,
                "password_hash": hash_password(settings.admin_password),
                "role": "admin",
            }
        )

    _atomic_modify(_modify)


def generate_invite(role: str = "user", count: int = 1) -> list[str]:
    codes: list[str] = []

    def _modify(data):
        for _ in range(count):
            code = f"INV-{secrets.token_urlsafe(9).upper()}"
            data["invites"].append(
                {
                    "code": code,
                    "used": False,
                    "role": role,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            codes.append(code)

    _atomic_modify(_modify)
    return codes


def list_users_and_invites() -> dict:
    data = _load_data()
    return {
        "users": [{"username": u["username"], "role": u["role"]} for u in data["users"]],
        "invites": [
            {
                "code": i["code"],
                "role": i["role"],
                "used": i["used"],
                "used_by": i.get("used_by"),
                "created_at": i.get("created_at"),
            }
            for i in data["invites"]
        ],
    }
