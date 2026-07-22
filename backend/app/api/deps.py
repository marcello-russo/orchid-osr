"""Authentication dependencies: require_role, get_current_user."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.core.auth import decode_jwt


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
        if role == "superadmin":
            return current_user
        if role == min_role:
            return current_user
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    return _check
