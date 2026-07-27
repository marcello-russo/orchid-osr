"""Authentication dependencies: require_role, get_current_user."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.core.auth import ROLE_HIERARCHY, decode_jwt


async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    token = auth[7:]
    payload = decode_jwt(token)
    return {"username": payload["sub"], "role": payload["role"]}


def require_role(min_role: str):
    def _check(current_user: dict = Depends(get_current_user)):
        role = current_user.get("role", "")
        user_level = ROLE_HIERARCHY.get(role, -1)
        min_level = ROLE_HIERARCHY.get(min_role, -1)
        if user_level >= min_level:
            return current_user
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    return _check
