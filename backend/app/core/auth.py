import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
import jwt
from fastapi import HTTPException, status

from app.config import settings

USERS_PATH = Path(os.environ.get("USERS_PATH", "data/users.json"))
_HF_DATASET_REPO = "marcellorusso/orchid-ncd-dataset"
JWT_ALGORITHM = "HS256"


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


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_jwt(username: str, role: str) -> str:
    payload = {"sub": username, "role": role}
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


def register_user(invite_code: str, username: str, password: str) -> dict:
    data = _load_data()

    invite = next((i for i in data["invites"] if i["code"] == invite_code), None)
    if not invite:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid invite code")
    if invite["used"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invite code already used")
    if any(u["username"] == username for u in data["users"]):
        raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")

    invite["used"] = True
    role = invite["role"]
    data["users"].append(
        {
            "username": username,
            "password_hash": hash_password(password),
            "role": role,
        }
    )
    _save_data(data)
    return {"username": username, "role": role}


def login_user(username: str, password: str) -> dict:
    data = _load_data()
    user = next((u for u in data["users"] if u["username"] == username), None)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    return {"username": user["username"], "role": user["role"]}


def ensure_admin_exists() -> None:
    data = _load_data()
    if not any(u["role"] == "admin" for u in data["users"]):
        data["users"].append(
            {
                "username": settings.admin_username,
                "password_hash": hash_password(settings.admin_password),
                "role": "admin",
            }
        )
        _save_data(data)


def generate_invite(role: str = "user", count: int = 1) -> list[str]:
    data = _load_data()
    codes = []
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
    _save_data(data)
    return codes


def list_users_and_invites() -> dict:
    data = _load_data()
    return {
        "users": [{"username": u["username"], "role": u["role"]} for u in data["users"]],
        "invites": [{"code": i["code"], "used": i["used"], "role": i["role"]} for i in data["invites"]],
    }
