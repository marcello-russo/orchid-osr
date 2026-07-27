# Auth System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add invite-based authentication with three roles (public/user/admin) to OrchID OSR.

**Architecture:** Backend uses a JSON file for users/invites, bcrypt for passwords, JWT for sessions, and FastAPI dependency injection for route protection. Frontend uses Zustand (already in the project) for auth state and adds a login/register page.

**Tech Stack:** FastAPI, JWT (PyJWT), bcrypt, React (shadcn/ui), Zustand

---

### Task 1: Backend dependencies

**Files:**
- Modify: `resnet18/backend/requirements.txt`

- [ ] **Step 1: Add pyjwt and bcrypt to requirements**

```
pyjwt==2.10.1
bcrypt==4.2.1
```

- [ ] **Step 2: Install**

Run: `cd resnet18/backend && pip install pyjwt bcrypt`

- [ ] **Step 3: Commit**

```bash
git add resnet18/backend/requirements.txt
git commit -m "feat(auth): add pyjwt and bcrypt dependencies"
```

---

### Task 2: Backend config

**Files:**
- Modify: `resnet18/backend/app/config.py`
- Modify: `resnet18/.env`

- [ ] **Step 1: Add JWT_SECRET and admin credentials to Settings**

In `config.py`, add after `hf_token`:

```python
jwt_secret: str = ""
admin_username: str = "admin"
admin_password: str = "admin"
```

- [ ] **Step 2: Add values to .env**

In `resnet18/.env`, add:

```
JWT_SECRET=***RIMOSSO***
ADMIN_USERNAME=marcello
ADMIN_PASSWORD=<set-a-password>
```

- [ ] **Step 3: Commit**

```bash
git add resnet18/backend/app/config.py resnet18/.env
git commit -m "feat(auth): add JWT and admin config to settings"
```

---

### Task 3: Backend auth core — users.json management, password hashing, JWT

**Files:**
- Create: `resnet18/backend/app/core/auth.py`

- [ ] **Step 1: Write `app/core/auth.py`**

```python
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
import jwt
from fastapi import HTTPException, status

from app.config import settings

USERS_PATH = Path("users.json")
JWT_ALGORITHM = "HS256"


def _load_data() -> dict:
    if USERS_PATH.exists():
        return json.loads(USERS_PATH.read_text())
    return {"invites": [], "users": []}


def _save_data(data: dict) -> None:
    USERS_PATH.write_text(json.dumps(data, indent=2, default=str))


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
    data["users"].append({
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
    })
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
        data["users"].append({
            "username": settings.admin_username,
            "password_hash": hash_password(settings.admin_password),
            "role": "admin",
        })
        _save_data(data)


def generate_invite(role: str = "user", count: int = 1) -> list[str]:
    data = _load_data()
    codes = []
    for _ in range(count):
        code = f"INV-{secrets.token_urlsafe(9).upper()}"
        data["invites"].append({
            "code": code,
            "used": False,
            "role": role,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        codes.append(code)
    _save_data(data)
    return codes


def list_users_and_invites() -> dict:
    data = _load_data()
    return {
        "users": [{"username": u["username"], "role": u["role"]} for u in data["users"]],
        "invites": [{"code": i["code"], "used": i["used"], "role": i["role"]} for i in data["invites"]],
    }
```

- [ ] **Step 2: Commit**

```bash
git add resnet18/backend/app/core/auth.py
git commit -m "feat(auth): add auth core with JWT, bcrypt, users.json management"
```

---

### Task 4: Backend auth routes

**Files:**
- Create: `resnet18/backend/app/api/routes/auth.py`

- [ ] **Step 1: Write auth routes**

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.auth import (
    register_user,
    login_user,
    create_jwt,
    decode_jwt,
)

router = APIRouter(prefix="/auth", tags=["auth"])


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
def login(body: LoginRequest):
    user = login_user(body.username, body.password)
    token = create_jwt(user["username"], user["role"])
    return {"token": token, "user": user}


@router.get("/me")
def me(token_data: dict = Depends(lambda: ...)):
    from fastapi import Request
    from fastapi import HTTPException, status

    async def get_me(request: Request):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
        return decode_jwt(auth[7:])

    return get_me(Request)
```

Wait, the `me` endpoint is wrong. Let me fix:

```python
from fastapi import APIRouter, Depends, Request
from fastapi import HTTPException, status
from pydantic import BaseModel

from app.core.auth import register_user, login_user, create_jwt, decode_jwt

router = APIRouter(prefix="/auth", tags=["auth"])


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
def login(body: LoginRequest):
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
```

- [ ] **Step 2: Register router in server.py**

In `server.py`, add import and include:

```python
from app.api.routes.auth import router as auth_router
...
app.include_router(auth_router)
```

Also call `ensure_admin_exists()` at startup — add after other startup logic:

```python
from app.core.auth import ensure_admin_exists
...
@app.on_event("startup")
async def startup():
    ensure_admin_exists()
    ...
```

- [ ] **Step 3: Commit**

```bash
git add resnet18/backend/app/api/routes/auth.py resnet18/backend/server.py
git commit -m "feat(auth): add /auth/register, /auth/login, /auth/me endpoints"
```

---

### Task 5: Backend auth middleware — get_current_user, require_role

**Files:**
- Modify: `resnet18/backend/app/api/deps.py`

- [ ] **Step 1: Add get_current_user and require_role**

In `deps.py`, add:

```python
from fastapi import Depends, HTTPException, status, Request
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
```

- [ ] **Step 2: Commit**

```bash
git add resnet18/backend/app/api/deps.py
git commit -m "feat(auth): add get_current_user and require_role middleware"
```

---

### Task 6: Backend — protect endpoints

**Files:**
- Modify: `resnet18/backend/app/api/routes/dataset.py`
- Modify: `resnet18/backend/app/api/routes/classify.py`
- Modify: `resnet18/backend/app/api/routes/experiments.py`
- Modify: `resnet18/backend/app/api/routes/embeddings.py`
- Modify: `resnet18/backend/app/api/routes/species.py`
- Modify: `resnet18/backend/app/api/routes/comparison.py`
- Modify: `resnet18/backend/app/api/routes/ncd.py`

For each route file, add `from app.api.deps import require_role` and add `Depends(require_role(...))` to protected endpoints per the access matrix.

**Access matrix (protected only — public endpoints need no change):**

| Endpoint | File | Protection |
|---|---|---|
| `POST /classify/explain` | classify.py | `require_role("user")` |
| `GET /api/ncd/*/results` | ncd.py | `require_role("user")` |
| `GET /api/embeddings` | embeddings.py | `require_role("user")` |
| `GET /api/dataset/classes` | dataset.py | `require_role("user")` |
| `GET /api/dataset/classes/{name}/images` | dataset.py | `require_role("user")` |
| `GET /api/image/{class}/{file}` | dataset.py | `require_role("admin")` |
| `GET /api/dataset/images/{split}/{class}/{file}` | dataset.py | `require_role("admin")` |
| `GET /classify/cache` | classify.py | `require_role("admin")` |
| `POST /classify/cache/download*` | classify.py | `require_role("admin")` |
| `DELETE /classify/cache*` | classify.py | `require_role("admin")` |

Example pattern for a protected endpoint:

```python
@router.get("/api/dataset/classes")
async def list_classes(user=Depends(require_role("user"))):
    ...
```

- [ ] **Step 1: Classify routes** — add Depends to explain and cache endpoints

- [ ] **Step 2: Dataset routes** — add Depends to classes, images endpoints

- [ ] **Step 3: Embeddings, NCD routes** — add Depends

- [ ] **Step 4: Verify no endpoint was missed**

- [ ] **Step 5: Commit**

```bash
git add resnet18/backend/app/api/routes/classify.py resnet18/backend/app/api/routes/dataset.py
git add resnet18/backend/app/api/routes/embeddings.py resnet18/backend/app/api/routes/ncd.py
git commit -m "feat(auth): protect endpoints with role-based access"
```

---

### Task 7: Backend CLI — user management commands

**Files:**
- Modify: `resnet18/backend/app/cli/__init__.py`
- (existing CLI argument parsing)

- [ ] **Step 1: Add CLI arguments for user management**

In `resnet18/backend/cli.py`, add new arguments:

```python
parser.add_argument("--add-user", nargs="?", const=None, help="Create user directly (use with --username --password --role)")
parser.add_argument("--username", help="Username for --add-user")
parser.add_argument("--password", help="Password for --add-user")
parser.add_argument("--role", default="user", choices=["user", "admin"], help="Role for --add-user")
parser.add_argument("--gen-invites", nargs="?", type=int, const=1, help="Generate N invite tokens")
parser.add_argument("--list-users", action="store_true", help="List users and invites")
```

And handler logic in main:

```python
from app.core.auth import register_user, generate_invite, list_users_and_invites, ensure_admin_exists

ensure_admin_exists()

if args.list_users:
    data = list_users_and_invites()
    print("=== Users ===")
    for u in data["users"]:
        print(f"  {u['username']} ({u['role']})")
    print("\n=== Invites ===")
    for i in data["invites"]:
        status = "used" if i["used"] else "available"
        print(f"  {i['code']} [{status}] ({i['role']})")
    return

if args.gen_invites is not None:
    codes = generate_invite(count=args.gen_invites)
    for c in codes:
        print(c)
    return

if args.add_user is not None:
    if not args.password:
        print("--password is required with --add-user")
        return
    username = args.add_user or args.username
    if not username:
        print("Specify username with --add-user <name> or --username")
        return
    # Direct add: bypass invite, create user immediately
    from app.core.auth import hash_password, _load_data, _save_data
    data = _load_data()
    if any(u["username"] == username for u in data["users"]):
        print(f"User '{username}' already exists")
        return
    data["users"].append({
        "username": username,
        "password_hash": hash_password(args.password),
        "role": args.role,
    })
    _save_data(data)
    print(f"User '{username}' created with role '{args.role}'")
    return
```

- [ ] **Step 2: Test the CLI commands**

```bash
cd resnet18/backend && python cli.py --list-users
cd resnet18/backend && python cli.py --gen-invites 3
cd resnet18/backend && python cli.py --add-user testuser --password test123 --role user
```

- [ ] **Step 3: Commit**

```bash
git add resnet18/backend/cli.py
git commit -m "feat(auth): add CLI commands for user and invite management"
```

---

### Task 8: Frontend — extend Zustand auth store

**Files:**
- Modify: `resnet18/frontend/src/lib/store/useAppStore.ts`
- Modify: `resnet18/frontend/src/lib/types.ts`

- [ ] **Step 1: Add auth types**

In `types.ts`:

```typescript
export interface AuthUser {
  username: string;
  role: "admin" | "user";
}

export interface AuthState {
  token: string | null;
  user: AuthUser | null;
}
```

- [ ] **Step 2: Extend Zustand store**

In `useAppStore.ts`, add to the store interface:

```typescript
interface AuthSlice {
  token: string | null;
  user: AuthUser | null;
  setAuth: (token: string, user: AuthUser) => void;
  logout: () => void;
}
```

Add to the store creation:

```typescript
token: null,
user: null,
setAuth: (token: string, user: AuthUser) => set({ token, user }),
logout: () => set({ token: null, user: null }),
```

- [ ] **Step 3: Commit**

```bash
git add resnet18/frontend/src/lib/store/useAppStore.ts resnet18/frontend/src/lib/types.ts
git commit -m "feat(auth): add auth state to Zustand store"
```

---

### Task 9: Frontend — API layer with auth headers

**Files:**
- Modify: `resnet18/frontend/src/lib/api.ts`

- [ ] **Step 1: Add auth header to all API calls**

Import the store and add a helper:

```typescript
import { useAppStore } from "./store/useAppStore";

function getAuthHeaders(): Record<string, string> {
  const token = useAppStore.getState().token;
  if (token) return { Authorization: `Bearer ${token}` };
  return {};
}
```

Add `getAuthHeaders()` to every request (except auth endpoints). 

Example for classify:

```typescript
const headers = { "Content-Type": "application/json", ...getAuthHeaders() };
```

Use a wrapper pattern — create `apiFetch`:

```typescript
async function apiFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const headers = {
    ...options.headers,
    ...getAuthHeaders(),
  };
  // Skip auth header for auth endpoints
  if (url.startsWith("/auth/")) {
    delete headers.Authorization;
  }
  return fetch(url, { ...options, headers });
}
```

Replace all direct `fetch()` calls with `apiFetch()`.

- [ ] **Step 2: Add auth API functions**

```typescript
export async function login(username: string, password: string): Promise<{ token: string; user: AuthUser }> {
  const res = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function register(inviteCode: string, username: string, password: string): Promise<{ token: string; user: AuthUser }> {
  const res = await apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify({ invite_code: inviteCode, username, password }),
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchMe(): Promise<{ username: string; role: string }> {
  const res = await apiFetch("/auth/me");
  if (!res.ok) throw new Error("Not authenticated");
  return res.json();
}
```

- [ ] **Step 3: Commit**

```bash
git add resnet18/frontend/src/lib/api.ts
git commit -m "feat(auth): add auth headers and login/register API calls"
```

---

### Task 10: Frontend — Missing shadcn/ui components (Input, Label)

**Files:**
- Create: `resnet18/frontend/src/components/ui/input.tsx`
- Create: `resnet18/frontend/src/components/ui/label.tsx`

- [ ] **Step 1: Create Input component**

```tsx
import * as React from "react"
import { cn } from "@/lib/utils"

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        data-slot="input"
        className={cn(
          "flex h-9 w-full rounded-lg border border-input bg-transparent px-3 py-1 text-sm shadow-xs transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
```

- [ ] **Step 2: Create Label component**

```tsx
import * as React from "react"
import { cn } from "@/lib/utils"

const Label = React.forwardRef<HTMLLabelElement, React.ComponentProps<"label">>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      data-slot="label"
      className={cn(
        "text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
)
Label.displayName = "Label"

export { Label }
```

- [ ] **Step 3: Commit**

```bash
git add resnet18/frontend/src/components/ui/input.tsx resnet18/frontend/src/components/ui/label.tsx
git commit -m "chore(ui): add Input and Label shadcn components"
```

---

### Task 11: Frontend — Auth page (login/register)

**Files:**
- Create: `resnet18/frontend/src/pages/AuthPage.tsx`

- [ ] **Step 1: Write the AuthPage component**

```tsx
import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAppStore } from "../lib/store/useAppStore";
import { login, register } from "../lib/api";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Card, CardContent, CardHeader } from "../components/ui/card";
import { AlertCircle, Eye, EyeOff, Loader2, User, Key } from "lucide-react";

export default function AuthPage() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [inviteCode, setInviteCode] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { setAuth } = useAppStore();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as any)?.from || "/";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (mode === "register" && password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    setLoading(true);
    try {
      const result = mode === "login"
        ? await login(username, password)
        : await register(inviteCode, username, password);
      setAuth(result.token, result.user);
      navigate(from, { replace: true });
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const switchMode = () => {
    setMode(mode === "login" ? "register" : "login");
    setError("");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center pb-2">
          <div className="mb-4">
            <span className="text-primary font-bold text-xl">Orch</span>
            <span className="text-foreground font-bold text-xl">ID</span>
            <span className="text-muted-foreground font-bold text-xs ml-0.5">NCD</span>
          </div>
          <div className="flex rounded-lg bg-muted p-1">
            <button
              onClick={() => switchMode()}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
                mode === "login" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Accedi
            </button>
            <button
              onClick={() => switchMode()}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
                mode === "register" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Registrati
            </button>
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "register" && (
              <div className="space-y-2">
                <Label htmlFor="invite">Codice invito</Label>
                <div className="relative">
                  <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="invite"
                    value={inviteCode}
                    onChange={(e) => setInviteCode(e.target.value.toUpperCase())}
                    className="pl-10 font-mono"
                    placeholder="INV-..."
                    required
                  />
                </div>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="pl-10"
                  placeholder="marcello"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10 pr-10"
                  placeholder="••••••••"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {mode === "register" && (
              <div className="space-y-2">
                <Label htmlFor="confirm">Conferma password</Label>
                <div className="relative">
                  <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="confirm"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="pl-10"
                    placeholder="••••••••"
                    required
                  />
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4" />
                <span>{error}</span>
              </div>
            )}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {mode === "login" ? "Accedi" : "Registrati"}
            </Button>

            <p className="text-center text-sm text-muted-foreground">
              {mode === "login" ? (
                <>Non hai un account? <button type="button" onClick={switchMode} className="text-primary hover:underline font-medium">Registrati</button></>
              ) : (
                <>Hai già un account? <button type="button" onClick={switchMode} className="text-primary hover:underline font-medium">Accedi</button></>
              )}
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add resnet18/frontend/src/pages/AuthPage.tsx
git commit -m "feat(auth): add login/register page with invite code support"
```

---

### Task 12: Frontend — ProtectedRoute component

**Files:**
- Create: `resnet18/frontend/src/components/auth/ProtectedRoute.tsx`

- [ ] **Step 1: Write ProtectedRoute**

```tsx
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAppStore } from "../../lib/store/useAppStore";

interface ProtectedRouteProps {
  role?: "user" | "admin";
}

export default function ProtectedRoute({ role = "user" }: ProtectedRouteProps) {
  const { token, user } = useAppStore();
  const location = useLocation();

  if (!token || !user) {
    return <Navigate to="/auth" state={{ from: location }} replace />;
  }

  if (role === "admin" && user.role !== "admin") {
    return <Navigate to="/" replace />;
  }

  return <Outlet />;
}
```

- [ ] **Step 2: Commit**

```bash
git add resnet18/frontend/src/components/auth/ProtectedRoute.tsx
mkdir -p resnet18/frontend/src/components/auth
git add resnet18/frontend/src/components/auth/ProtectedRoute.tsx
git commit -m "feat(auth): add ProtectedRoute component with role check"
```

---

### Task 13: Frontend — integrate auth into App.tsx and sidebar

**Files:**
- Modify: `resnet18/frontend/src/App.tsx`
- Modify: `resnet18/frontend/src/components/layout/Sidebar.tsx` or `Header.tsx`

- [ ] **Step 1: Add `/auth` route and wrap protected routes**

In `App.tsx`, import `ProtectedRoute` and add:

```tsx
import AuthPage from "./pages/AuthPage";
import ProtectedRoute from "./components/auth/ProtectedRoute";
```

Add the auth route:

```tsx
<Route path="/auth" element={<AuthPage />} />
```

Wrap protected page routes:

```tsx
<Route element={<ProtectedRoute role="user" />}>
  {/* Pages that require login */}
</Route>

<Route element={<ProtectedRoute role="admin" />}>
  {/* Pages that require admin */}
</Route>
```

- [ ] **Step 2: Add user status to header/sidebar (logout button)**

In `Header.tsx`, add after the theme toggle:

```tsx
import { useAppStore } from "../../lib/store/useAppStore";
import { useNavigate } from "react-router-dom";

const { token, user, logout } = useAppStore();
const navigate = useNavigate();

// In the JSX, add:
{token ? (
  <div className="flex items-center gap-2">
    <span className="text-sm text-muted-foreground">{user?.username}</span>
    <button
      onClick={() => { logout(); navigate("/"); }}
      className="text-sm text-muted-foreground hover:text-foreground transition-colors"
    >
      Logout
    </button>
  </div>
) : (
  <button
    onClick={() => navigate("/auth")}
    className="text-sm text-primary hover:underline"
  >
    Accedi
  </button>
)}
```

- [ ] **Step 3: Commit**

```bash
git add resnet18/frontend/src/App.tsx resnet18/frontend/src/components/layout/Header.tsx
git commit -m "feat(auth): integrate auth routes and user menu in header"
```

---

### Task 14: Frontend — i18n auth translations

**Files:**
- Modify: `resnet18/frontend/src/locales/en.json`
- Modify: `resnet18/frontend/src/locales/it.json`

- [ ] **Step 1: Add EN keys**

In `en.json`:

```json
{
  "auth": {
    "login": "Login",
    "register": "Register",
    "inviteCode": "Invite code",
    "username": "Username",
    "password": "Password",
    "confirmPassword": "Confirm password",
    "loginButton": "Sign in",
    "registerButton": "Sign up",
    "noAccount": "Don't have an account? Register",
    "hasAccount": "Already have an account? Sign in",
    "logout": "Logout",
    "errors": {
      "passwordsDontMatch": "Passwords do not match",
      "invalidInvite": "Invalid or expired invite code",
      "usernameTaken": "Username already taken",
      "invalidCredentials": "Invalid username or password"
    }
  }
}
```

- [ ] **Step 2: Add IT keys**

In `it.json`:

```json
{
  "auth": {
    "login": "Accedi",
    "register": "Registrati",
    "inviteCode": "Codice invito",
    "username": "Username",
    "password": "Password",
    "confirmPassword": "Conferma password",
    "loginButton": "Accedi",
    "registerButton": "Registrati",
    "noAccount": "Non hai un account? Registrati",
    "hasAccount": "Hai già un account? Accedi",
    "logout": "Esci",
    "errors": {
      "passwordsDontMatch": "Le password non coincidono",
      "invalidInvite": "Codice invito non valido o scaduto",
      "usernameTaken": "Username già in uso",
      "invalidCredentials": "Username o password non validi"
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add resnet18/frontend/src/locales/en.json resnet18/frontend/src/locales/it.json
git commit -m "feat(auth): add auth translations (EN/IT)"
```

---

## Self-Review

- [ ] Spec coverage: every requirement from the design doc has a corresponding task
- [ ] No placeholders (TBD, TODO, etc.)
- [ ] Type consistency across tasks
- [ ] File paths are exact and correct
