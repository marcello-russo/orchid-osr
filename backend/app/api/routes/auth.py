from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.auth import create_jwt, decode_jwt, login_user, register_user

router = APIRouter(prefix="/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)


class RegisterRequest(BaseModel):
    invite_code: str
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
def register(body: RegisterRequest):
    user = register_user(body.invite_code, body.username, body.password)
    token = create_jwt(user["username"], user["role"])
    return {"token": token, "user": user}


@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
    user = login_user(body.username, body.password)
    token = create_jwt(user["username"], user["role"])
    return {"token": token, "user": user}


@router.get("/me")
def me(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    payload = decode_jwt(auth[7:])
    return {"username": payload["sub"], "role": payload["role"]}
