"""Authentication dependencies: require_role, get_current_user."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from app.core.auth import ROLE_HIERARCHY, decode_jwt


async def get_current_user(request: Request) -> dict[str, Any]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    token = auth[7:]
    payload = decode_jwt(token)
    return {"username": payload["sub"], "role": payload["role"]}


async def get_current_user_optional(request: Request) -> dict[str, Any] | None:
    """Utente se il token c'e' ed e' valido, altrimenti None.

    Serve alle pagine pubbliche che mostrano qualcosa in piu' a chi ha un
    account (anteprima del dataset, classificazione).
    """
    try:
        return await get_current_user(request)
    except HTTPException:
        return None


def require_role(min_role: str) -> Callable[..., dict[str, Any]]:
    def _check(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        role = current_user.get("role", "")
        user_level = ROLE_HIERARCHY.get(role, -1)
        min_level = ROLE_HIERARCHY.get(min_role, -1)
        if user_level >= min_level:
            return current_user
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    return _check
