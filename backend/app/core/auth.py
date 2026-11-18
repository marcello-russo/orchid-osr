import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import bcrypt
import jwt
from fastapi import HTTPException, status

from app.config import settings
from app.core import auth_store
from app.core.db import get_db

logger = logging.getLogger(__name__)

# Solo per l'import una-tantum dei dati v0.3.x: dopo la migrazione il file non
# viene più letto (il marker sta in auth_meta) né scritto.
USERS_PATH = Path(os.environ.get("USERS_PATH", "data/users.json"))
JWT_ALGORITHM = "HS256"
_LOGIN_ATTEMPTS: OrderedDict[str, int] = OrderedDict()
LOGIN_ATTEMPTS_CACHE: dict[str, int] = {}
_CACHE_LAST_SYNC: float = 0.0
MAX_LOGIN_ATTEMPTS = 5
_MAX_TRACKED_USERS = 1000
_LOGIN_LOCK = threading.Lock()
MAX_INVITE_HOURS = 720  # 30 giorni
MAX_GUEST_PASS_HOURS = 168  # 7 giorni

ROLE_HIERARCHY = {"user": 0, "admin": 1, "superadmin": 2}


def _initialize() -> None:
    auth_store.initialize(USERS_PATH)


def initialize_auth() -> None:
    """Prepara lo storage auth all'avvio (import legacy una-tantum incluso).

    Chiamata esplicita dal lifespan: un `users.json` malformato deve fermare il
    container con un errore chiaro, non diventare un 500 al primo login.
    """
    _initialize()


def _load_attempts() -> dict[str, int]:
    global _CACHE_LAST_SYNC
    if time.time() - _CACHE_LAST_SYNC < 5.0:
        return LOGIN_ATTEMPTS_CACHE
    try:
        with closing(get_db()) as conn:
            rows = conn.execute("SELECT username, attempts FROM login_attempts").fetchall()
        LOGIN_ATTEMPTS_CACHE.clear()
        LOGIN_ATTEMPTS_CACHE.update({r[0]: r[1] for r in rows})
        _CACHE_LAST_SYNC = time.time()
    except sqlite3.Error:
        logger.warning("Tentativi di login non leggibili dal DB", exc_info=True)
    return LOGIN_ATTEMPTS_CACHE


def _save_attempts(username: str, count: int) -> None:
    try:
        with closing(get_db()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO login_attempts (username, attempts, last_attempt) VALUES (?, ?, ?)",
                (username, count, time.time()),
            )
            conn.commit()
    except sqlite3.Error:
        logger.warning("Tentativi di login non salvati nel DB", exc_info=True)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_jwt(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "aud": "orchid-osr",
        "iss": "orchid-osr-api",
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
            audience="orchid-osr",
            issuer="orchid-osr-api",
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


def _parse_deadline(value: Any) -> datetime | None:
    """Interpreta una scadenza ISO; None se assente, ValueError se malformata."""
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _invite_error(invite: dict[str, Any]) -> str | None:
    """Ritorna il motivo di rifiuto di un invito, o None se utilizzabile.

    Fail-closed: una data malformata rende l'invito invalido, non eterno.
    """
    if invite.get("revoked"):
        return "Invite code revoked"
    try:
        expires = _parse_deadline(invite.get("expires_at"))
    except ValueError:
        return "Invite code invalid"
    if expires is not None and expires <= datetime.now(UTC):
        return "Invite code expired"
    uses = int(invite.get("uses", 0))
    max_uses = int(invite.get("max_uses", 1))
    if invite.get("used") or uses >= max_uses:
        return "Invite code already used"
    return None


def register_user(invite_code: str, username: str, password: str) -> dict[str, Any]:
    _initialize()
    # bcrypt costa ~250ms: si calcola prima di prendere il lock di scrittura.
    hashed = hash_password(password)
    with auth_store.transaction() as conn:
        invite = auth_store.find_invite(conn, invite_code)
        if invite is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid invite code")
        err = _invite_error(invite)
        if err:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, err)
        if auth_store.find_user(conn, username):
            raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")

        role = str(invite["role"])
        auth_store.insert_user(conn, username, hashed, role)
        auth_store.consume_invite(conn, invite_code, username)
    return {"username": username, "role": role}


def login_user(username: str, password: str) -> dict[str, Any]:
    _initialize()
    _load_attempts()
    with _LOGIN_LOCK:
        if LOGIN_ATTEMPTS_CACHE.get(username, 0) >= MAX_LOGIN_ATTEMPTS:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Account temporarily locked. Try again later.")

    user = auth_store.get_user(username)
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


def ensure_superadmin_exists() -> None:
    username = settings.superadmin_username
    password = settings.superadmin_password
    if not username or not password:
        logger.warning("SUPERADMIN_USERNAME/PASSWORD non configurati — skip bootstrap")
        return
    _initialize()
    hashed = hash_password(password)
    with auth_store.transaction() as conn:
        auth_store.upsert_user(conn, username, hashed, "superadmin")
    logger.info("Superadmin %s allineato", username)


def generate_invite(
    role: str = "user",
    count: int = 1,
    *,
    expires_in_hours: int = 0,
    max_uses: int = 1,
    label: str = "",
    code: str | None = None,
) -> list[str]:
    """Crea inviti. expires_in_hours=0 → senza scadenza; max_uses=1 → monouso."""
    if max_uses < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "max_uses must be >= 1")
    if expires_in_hours < 0 or expires_in_hours > MAX_INVITE_HOURS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"expires_in_hours must be 0..{MAX_INVITE_HOURS}")
    if role not in ("user", "admin"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid role")
    if code and count > 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Custom code requires count=1")

    expires_at = (datetime.now(UTC) + timedelta(hours=expires_in_hours)).isoformat() if expires_in_hours > 0 else None
    _initialize()
    codes: list[str] = []
    with auth_store.transaction() as conn:
        for _ in range(count):
            new_code = code or f"INV-{secrets.token_urlsafe(9).upper()}"
            if auth_store.find_invite(conn, new_code):
                raise HTTPException(status.HTTP_409_CONFLICT, "Invite code already exists")
            auth_store.insert_invite(
                conn,
                {
                    "code": new_code,
                    "role": role,
                    "used": False,
                    "used_by": None,
                    "created_at": datetime.now(UTC).isoformat(),
                    "expires_at": expires_at,
                    "max_uses": max_uses,
                    "uses": 0,
                    "label": label,
                    "revoked": False,
                },
            )
            codes.append(new_code)
    return codes


def revoke_invite(code: str) -> None:
    """Revoca un invito (kill-switch manuale). Idempotente."""
    _initialize()
    with auth_store.transaction() as conn:
        if not auth_store.set_invite_revoked(conn, code):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Invite code not found")


GUEST_USERNAME = "ospite"


def ensure_guest_user() -> None:
    """Crea l'utente ospite condiviso (ruolo user) se manca."""
    _initialize()
    if auth_store.get_user(GUEST_USERNAME):
        return
    hashed = hash_password(secrets.token_urlsafe(32))
    with auth_store.transaction() as conn:
        if not auth_store.find_user(conn, GUEST_USERNAME):
            auth_store.insert_user(conn, GUEST_USERNAME, hashed, "user")


def create_guest_pass(*, expires_in_hours: int = 24, label: str = "") -> dict[str, Any]:
    """Crea un passo ospite: token firmato a tempo che logga come ospite."""
    if expires_in_hours < 1 or expires_in_hours > MAX_GUEST_PASS_HOURS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"expires_in_hours must be 1..{MAX_GUEST_PASS_HOURS}",
        )
    _initialize()
    now = int(time.time())
    jti = secrets.token_urlsafe(16)
    payload = {
        "sub": GUEST_USERNAME,
        "role": "user",
        "purpose": "guest-pass",
        "jti": jti,
        "aud": "orchid-osr",
        "iss": "orchid-osr-api",
        "iat": now,
        "exp": now + expires_in_hours * 3600,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)
    record = {
        "jti": jti,
        "label": label,
        "created_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(hours=expires_in_hours)).isoformat(),
        "revoked": False,
    }

    ensure_guest_user()
    with auth_store.transaction() as conn:
        auth_store.insert_guest_pass(conn, record)
    return {"pass": token, **record}


def redeem_guest_pass(token: str) -> dict[str, str]:
    """Valida un passo ospite e restituisce le credenziali di sessione."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[JWT_ALGORITHM],
            audience="orchid-osr",
            issuer="orchid-osr-api",
        )
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired pass")
    if payload.get("purpose") != "guest-pass" or payload.get("sub") != GUEST_USERNAME:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid pass")

    _initialize()
    record = auth_store.get_guest_pass(str(payload.get("jti")))
    if record is None or record.get("revoked"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Pass revoked")
    try:
        expires = _parse_deadline(record.get("expires_at"))
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid pass") from None
    if expires is not None and expires <= datetime.now(UTC):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Pass expired")
    return {"username": GUEST_USERNAME, "role": "user"}


def revoke_guest_pass(jti: str) -> None:
    """Revoca un passo ospite (kill-switch manuale). Idempotente."""
    _initialize()
    with auth_store.transaction() as conn:
        if not auth_store.set_guest_pass_revoked(conn, jti):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Pass not found")


def list_users_and_invites() -> dict[str, Any]:
    _initialize()
    return {
        "users": auth_store.list_users(),
        "invites": [
            {
                "code": i["code"],
                "role": i["role"],
                "used": i["used"],
                "used_by": i.get("used_by"),
                "created_at": i.get("created_at"),
                "expires_at": i.get("expires_at"),
                "max_uses": i.get("max_uses", 1),
                "uses": i.get("uses", 0),
                "label": i.get("label", ""),
                "revoked": i.get("revoked", False),
            }
            for i in auth_store.list_invites()
        ],
    }


def delete_user(target_username: str, requester_username: str) -> None:
    if target_username == requester_username:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot delete yourself")

    _initialize()
    with auth_store.transaction() as conn:
        target = auth_store.find_user(conn, target_username)
        if not target:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

        requester = auth_store.find_user(conn, requester_username)
        if not requester:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Requester not found")

        target_level = ROLE_HIERARCHY.get(str(target["role"]), -1)
        requester_level = ROLE_HIERARCHY.get(str(requester["role"]), -1)
        if requester_level <= target_level:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Cannot delete a user with equal or higher role")

        auth_store.delete_user(conn, target_username)
        # `uses` va azzerato insieme a `used`: con il solo used=0 l'invito
        # resterebbe bruciato (uses >= max_uses).
        conn.execute(
            "UPDATE invites SET used = 0, uses = 0, used_by = NULL WHERE used_by = ?",
            (target_username,),
        )


def create_user(username: str, password: str, role: str = "user") -> None:
    """Crea un utente da CLI. 409 se esiste già."""
    if role not in ROLE_HIERARCHY:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid role")
    _initialize()
    hashed = hash_password(password)
    with auth_store.transaction() as conn:
        if auth_store.find_user(conn, username):
            raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")
        auth_store.insert_user(conn, username, hashed, role)
