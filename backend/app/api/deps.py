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
    if min_role == "admin":

        async def _check(user: dict = Depends(get_current_user)):
            if user["role"] != "admin":
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
            return user

        return _check
    else:

        async def _check(user: dict = Depends(get_current_user)):
            return user

        return _check
