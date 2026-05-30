# Auth System — OrchID

## Overview

Authentication and authorization system for OrchID. Three roles (public, registered user, admin) with invite-based registration. Zero database — all state in a JSON file. Minimal admin overhead: generate invite tokens, share them, done.

## Architecture

```
Backend: JSON file → JWT middleware → role-based route protection
Frontend: Login/Register page → Zustand token store → Auth header on API calls
```

### Dependencies (new)

- `pyjwt` (o `python-jose`) — JWT sign/verify
- `bcrypt` — password hashing

## Data Model

File: `resnet18/backend/users.json`

```json
{
  "invites": [
    {
      "code": "INV-aB3kL9mX2zQ7",
      "used": false,
      "role": "user",
      "created_at": "2026-05-26T12:00:00"
    }
  ],
  "users": [
    {
      "username": "marcello",
      "password_hash": "$2b$12$...",
      "role": "admin"
    }
  ]
}
```

- Admin account creato automaticamente al primo avvio se non esiste, con credenziali da `.env` o generate via CLI
- Invite token: 12 caratteri alfanumerici, `secrets.token_urlsafe(9)`, prefisso `INV-`
- Password hashate con bcrypt (cost 12)

## CLI Commands

Nuovi comandi in `cli.py`:

- `--add-user <username> --password <pw> --role user|admin` — crea account direttamente
- `--gen-invites [n]` — genera n token (default 1), stampa tutti
- `--list-users` — lista utenti e inviti con stato

## API Endpoints

### Auth

| Method | Path | Body | Response |
|--------|------|------|----------|
| POST | `/auth/register` | `{invite_code, username, password}` | `{token, user: {username, role}}` |
| POST | `/auth/login` | `{username, password}` | `{token, user: {username, role}}` |
| GET | `/auth/me` | — (Authorization header) | `{username, role}` |

### Registration flow

1. User sends `invite_code` + `username` + `password`
2. Server checks invite exists, not used, marks as used
3. Creates user, returns JWT

### Login flow

1. User sends `username` + `password`
2. Server verifies bcrypt hash
3. Returns JWT (payload: `{sub: username, role: user|admin}`)

## Middleware

```python
# app/core/auth.py
def get_current_user(request) -> dict: ...
def require_role(role: str) -> Callable: ...
```

- `get_current_user`: estrae JWT da `Authorization: Bearer <token>`, verifica firma, restituisce `{username, role}`
- `require_role("admin")`: decorator/dependency che controlla il ruolo, alza 403 se insufficiente

### Route protection

```python
# Endpoint pubblico
@app.get("/health")
async def health(): ...

# Endpoint user+
@app.get("/api/embeddings")
async def embeddings(user = Depends(require_role("user"))): ...

# Endpoint admin+
@app.get("/api/image/{class_name}/{filename}")
async def image(admin = Depends(require_role("admin"))): ...
```

`require_role("user")` accepta sia user che admin.
`require_role("admin")` accetta solo admin.

### Endpoint access matrix

| Endpoint | Pubblico | User | Admin |
|---|---|---|---|
| `GET /health` | ✅ | ✅ | ✅ |
| `POST /classify`, `/upload` | ✅ | ✅ | ✅ |
| `POST /classify/explain` | ❌ | ✅ | ✅ |
| `GET /api/experiments*` | ✅ | ✅ | ✅ |
| `GET /api/comparison*` | ✅ | ✅ | ✅ |
| `GET /api/species` | ✅ | ✅ | ✅ |
| `GET /api/ncd/*/figure/*` | ✅ | ✅ | ✅ |
| `GET /api/ncd/*/results` | ❌ | ✅ | ✅ |
| `GET /api/embeddings` | ❌ | ✅ | ✅ |
| `GET /api/dataset/classes` | ❌ | ✅ | ✅ |
| `GET /api/dataset/classes/{name}/images` | ❌ | ✅ | ✅ |
| `GET /api/image/{class}/{file}` | ❌ | ❌ | ✅ |
| `GET /api/dataset/images/{split}/{class}/{file}` | ❌ | ❌ | ✅ |
| `GET /classify/cache` | ❌ | ❌ | ✅ |
| `POST /classify/cache/download*` | ❌ | ❌ | ✅ |
| `DELETE /classify/cache*` | ❌ | ❌ | ✅ |
| `POST /auth/register`, `/auth/login` | ✅ | — | — |

## Frontend

### Auth page (`/auth`)

- Layout centrato full-height, sfondo `bg-background`
- Card `bg-card border`
- Logo "OrchID NCD" in alto (stesso stile sidebar)
- Tab toggle Login / Register con stile pill

**Login tab:**
- Campo username con icona
- Campo password con toggle mostra/nascondi
- Bottone "Accedi" (componente `Button` shadcn, `variant="default"`)
- Link "Non hai un account? Registrati" in `text-primary`

**Register tab:**
- Campo "Codice invito" — input monospace, prefisso `INV-`
- Campo username
- Campo password
- Campo conferma password
- Bottone "Registrati"
- Link "Hai già un account? Accedi"

**Comportamento:**
- Submit: bottone si disabilita + spinner
- Errori: messaggio sotto il campo specifico (colore `destructive`)
- Successo: salva token + redirect alla pagina precedente
- Tab switch: transizione fluida 150ms

### Auth state (Zustand)

Estendere `useAppStore`:

```ts
interface AuthState {
  token: string | null;
  user: { username: string; role: "admin" | "user" } | null;
  setAuth: (token: string, user: ...) => void;
  logout: () => void;
  isAuthenticated: boolean;
}
```

Già esiste `persist` middleware con localStorage. Aggiungere chiavi `token` e `user`.

### API layer (`api.ts`)

Aggiungere header `Authorization: Bearer <token>` a tutte le richieste eccetto `/auth/register` e `/auth/login`. Leggere token dallo store Zustand.

### Route protection

Componente `ProtectedRoute`:

```tsx
<Route element={<ProtectedRoute role="user" />}>
  <Route path="..." element={<SomePage />} />
</Route>
```

- Controlla `isAuthenticated` dallo store
- Se non autenticato → redirect `/auth`
- Se ruolo insufficiente (admin page da user) → 403 / redirect

Pagine protette (user+):
- ClassifyPage (explain)
- ResearchPage (NCD results, embeddings)
- DatasetPage (galley classi)

Pagine admin+:
- Cache management
- Dataset immagini

## Config

Nuovo campo in `.env`:

```
JWT_SECRET=<random-string>
ADMIN_USERNAME=marcello
ADMIN_PASSWORD=<your-password>
```

- `JWT_SECRET`: generato automaticamente se non presente (alla prima esecuzione viene salvato)
- `ADMIN_USERNAME`/`ADMIN_PASSWORD`: usati per creare admin al primo avvio se `users.json` non esiste

## Files da creare/modificare

### Nuovi file

- `backend/app/core/auth.py` — JWT utils, password hashing, middleware
- `backend/app/api/routes/auth.py` — `/auth/register`, `/auth/login`, `/auth/me`
- `backend/app/cli/user_manager.py` — CLI per add-user, gen-invite, list-users
- `frontend/src/pages/AuthPage.tsx` — login/register page
- `frontend/src/components/auth/ProtectedRoute.tsx` — route guard
- `frontend/src/components/auth/AuthGuard.tsx` — componente per proteggere singoli elementi UI

### File da modificare

- `backend/app/cli/__init__.py` — registra nuovi comandi
- `backend/server.py` — registra auth router, include middleware JWT (?)
- `backend/app/api/deps.py` — aggiungi dipendenze `get_current_user`, `require_role`
- `backend/requirements.txt` — aggiungi `pyjwt`, `bcrypt`
- `frontend/src/lib/api.ts` — aggiungi auth header
- `frontend/src/lib/store/useAppStore.ts` — estendi con auth state
- `frontend/src/lib/types.ts` — aggiungi tipi auth
- `frontend/src/App.tsx` — aggiungi route `/auth` e `ProtectedRoute`
- `frontend/src/locales/en.json` + `it.json` — chiavi auth
- `resnet18/.env` — aggiungi JWT_SECRET, ADMIN_USERNAME, ADMIN_PASSWORD
