import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.deps import require_role
from app.core.auth import (
    MAX_GUEST_PASS_HOURS,
    create_guest_pass,
    create_jwt,
    decode_jwt,
    delete_user,
    generate_invite,
    list_users_and_invites,
    login_user,
    redeem_guest_pass,
    register_user,
    revoke_guest_pass,
    revoke_invite,
)

router = APIRouter(prefix="/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)


# Un invito senza scadenza è una credenziale dimenticata in un log: l'API ne
# impone sempre una (72h di default, 30 giorni al massimo). "Per sempre" resta
# possibile solo dal CLI, che è un'azione locale e consapevole.
DEFAULT_INVITE_HOURS = 72
MAX_INVITE_HOURS = 720


class GenerateInviteRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=10, description="Numero di inviti da generare (1-10)")
    role: str = Field(default="user", pattern=r"^(user|admin)$")
    expires_in_hours: int = Field(default=DEFAULT_INVITE_HOURS, ge=1, le=MAX_INVITE_HOURS)


class DefensePassRequest(BaseModel):
    """Passo multiuso a tempo (es. discussione): scade da solo, revocabile a mano.

    Solo ruolo user: un passo condiviso non deve mai conferire poteri admin.
    """

    role: Literal["user"] = "user"
    max_uses: int = Field(default=25, ge=1, le=200)
    expires_in_hours: int = Field(default=DEFAULT_INVITE_HOURS, ge=1, le=MAX_INVITE_HOURS)
    label: str = Field(default="difesa", max_length=40)


class GuestPassRequest(BaseModel):
    """Crea un magic link ospite: chi lo apre entra subito, senza registrarsi."""

    expires_in_hours: int = Field(default=24, ge=1, le=MAX_GUEST_PASS_HOURS)
    label: str = Field(default="difesa", max_length=40)


class GuestEnterRequest(BaseModel):
    token: str = Field(min_length=10, max_length=500)


class RegisterRequest(BaseModel):
    invite_code: str
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


@router.post("/register")
@limiter.limit("10/minute")
def register(request: Request, body: RegisterRequest) -> dict[str, Any]:  # noqa: ARG001 — required by slowapi
    user = register_user(body.invite_code, body.username, body.password)
    token = create_jwt(user["username"], user["role"])
    return {"token": token, "user": user}


@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest) -> dict[str, Any]:
    user = login_user(body.username, body.password)
    token = create_jwt(user["username"], user["role"])
    return {"token": token, "user": user}


@router.get("/me")
def me(request: Request) -> dict[str, Any]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    payload = decode_jwt(auth[7:])
    return {"username": payload["sub"], "role": payload["role"]}


@router.get("/invites")
def list_invites(request: Request, admin: dict[str, Any] = Depends(require_role("admin"))) -> dict[str, Any]:
    return list_users_and_invites()


@router.post("/invites/generate")
def create_invites(
    request: Request, body: GenerateInviteRequest, admin: dict[str, Any] = Depends(require_role("admin"))
) -> dict[str, list[str]]:
    codes = generate_invite(
        role=body.role,
        count=body.count,
        expires_in_hours=body.expires_in_hours,
    )
    return {"codes": codes}


@router.post("/invites/defense-pass")
def create_defense_pass(
    request: Request, body: DefensePassRequest, admin: dict[str, Any] = Depends(require_role("admin"))
) -> dict[str, Any]:
    codes = generate_invite(
        role=body.role,
        count=1,
        expires_in_hours=body.expires_in_hours,
        max_uses=body.max_uses,
        label=body.label,
    )
    invites = list_users_and_invites()["invites"]
    record = next(i for i in invites if i["code"] == codes[0])
    return {
        "code": record["code"],
        "role": record["role"],
        "label": record["label"],
        "max_uses": record["max_uses"],
        "expires_at": record["expires_at"],
    }


@router.post("/invites/{code}/revoke")
def revoke_invite_endpoint(
    code: str, request: Request, admin: dict[str, Any] = Depends(require_role("admin"))
) -> dict[str, str]:
    revoke_invite(code)
    return {"revoked": code}


@router.post("/guest-pass")
def create_guest_pass_endpoint(
    request: Request, body: GuestPassRequest, admin: dict[str, Any] = Depends(require_role("admin"))
) -> dict[str, Any]:
    record = create_guest_pass(expires_in_hours=body.expires_in_hours, label=body.label)
    return record


@router.post("/guest/revoke/{jti}")
def revoke_guest_pass_endpoint(
    jti: str, request: Request, admin: dict[str, Any] = Depends(require_role("admin"))
) -> dict[str, str]:
    revoke_guest_pass(jti)
    return {"revoked": jti}


@router.post("/guest/enter")
@limiter.limit("120/minute")
def guest_enter(request: Request, body: GuestEnterRequest) -> dict[str, Any]:  # noqa: ARG001 — required by slowapi
    user = redeem_guest_pass(body.token)
    token = create_jwt(user["username"], user["role"])
    return {"token": token, "user": user}


@router.delete("/users/{username}")
async def delete_user_endpoint(
    username: str,
    current_user: dict[str, Any] = Depends(require_role("admin")),
) -> dict[str, str]:
    """Delete a user. Superadmin can delete anyone, admin can only delete users."""
    delete_user(username, current_user["username"])
    return {"deleted": username}
