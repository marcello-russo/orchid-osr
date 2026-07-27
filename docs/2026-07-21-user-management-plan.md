# User Management Page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Invites page with full User Management panel at `/admin/users`. Introduce `superadmin` role from `.env`, add user deletion with role hierarchy.

**Architecture:** Backend changes (config, auth, routes, deps) then frontend (rename page, update sidebar, i18n).

**Tech Stack:** Python/FastAPI + React/TypeScript + Tailwind + shadcn/ui

---

### Task 1: Backend — configurare superadmin

**Files:**
- Modify: `backend/app/config.py`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `entrypoint.sh`

- [ ] **Step 1: Rimuovere admin e aggiungere superadmin in config.py**

In `backend/app/config.py`:

1. Sostituire:
```python
admin_username: str = ""
admin_password: str = ""
```
con:
```python
superadmin_username: str = ""
superadmin_password: str = ""
```

2. Rimuovere l'intero model_validator `_validate_admin_credentials` (le righe da `@model_validator(mode="after")` fino a `return self` prima della successiva funzione).

- [ ] **Step 2: Aggiornare .env.example**

In `.env.example`:
```
SUPERADMIN_USERNAME=admin
SUPERADMIN_PASSWORD=admin
```

- [ ] **Step 3: Aggiornare docker-compose.yml**

Nel servizio `backend`, sotto `environment:`, cambiare:
```yaml
      - ADMIN_USERNAME=${ADMIN_USERNAME}
      - ADMIN_PASSWORD=${ADMIN_PASSWORD}
```
in:
```yaml
      - SUPERADMIN_USERNAME=${SUPERADMIN_USERNAME}
      - SUPERADMIN_PASSWORD=${SUPERADMIN_PASSWORD}
```

- [ ] **Step 4: Aggiornare entrypoint.sh**

Cercare riferimenti a `ADMIN_USERNAME`/`ADMIN_PASSWORD` e sostituire con `SUPERADMIN_USERNAME`/`SUPERADMIN_PASSWORD`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/config.py .env.example docker-compose.yml entrypoint.sh
git commit -m "feat: SUPERADMIN_USERNAME/PASSWORD sostituisce ADMIN"
```

---

### Task 2: Backend — auth logic (superadmin bootstrap + delete_user)

**Files:**
- Modify: `backend/app/core/auth.py`

- [ ] **Step 1: Aggiungere ROLE_HIERARCHY e riscrivere ensure_admin_exists in ensure_superadmin_exists**

In `backend/app/core/auth.py`:

1. Aggiungere dopo gli import e prima della prima funzione:

```python
ROLE_HIERARCHY = {"user": 0, "admin": 1, "superadmin": 2}
```

2. Trovare `def ensure_admin_exists()` e rinominarla in `def ensure_superadmin_exists()`. Sostituire TUTTO il corpo con:

```python
def ensure_superadmin_exists() -> None:
    username = settings.superadmin_username
    password = settings.superadmin_password
    if not username or not password:
        logger.warning("SUPERADMIN_USERNAME/PASSWORD non configurati — skip bootstrap")
        return
    hashed = hash_password(password)
    def _modify(data):
        existing = next((u for u in data["users"] if u["username"] == username), None)
        if existing:
            existing["password_hash"] = hashed
            existing["role"] = "superadmin"
            logger.info("Superadmin %s aggiornato", username)
        else:
            data["users"].append({
                "username": username,
                "password_hash": hashed,
                "role": "superadmin",
            })
            logger.info("Superadmin %s creato", username)
    _atomic_modify(_modify)
```

- [ ] **Step 2: Aggiungere delete_user()**

Dopo `list_users_and_invites()`, aggiungere:

```python
def delete_user(target_username: str, requester_username: str) -> None:
    if target_username == requester_username:
        raise HTTPException(400, "Cannot delete yourself")

    def _modify(data):
        target = next((u for u in data["users"] if u["username"] == target_username), None)
        if not target:
            raise HTTPException(404, "User not found")

        requester = next((u for u in data["users"] if u["username"] == requester_username), None)
        if not requester:
            raise HTTPException(401, "Requester not found")

        target_level = ROLE_HIERARCHY.get(target["role"], -1)
        requester_level = ROLE_HIERARCHY.get(requester["role"], -1)

        if requester_level <= target_level:
            raise HTTPException(403, "Cannot delete a user with equal or higher role")

        data["users"] = [u for u in data["users"] if u["username"] != target_username]
        for invite in data["invites"]:
            if invite.get("used_by") == target_username:
                invite["used"] = False
                invite.pop("used_by", None)

    _atomic_modify(_modify)
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/core/auth.py
git commit -m "feat: ensure_superadmin_exists, delete_user con gerarchia ruoli"
```

---

### Task 3: Backend — require_role hierarchy + DELETE endpoint + bootstrap

**Files:**
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/api/routes/auth.py`
- Modify: `backend/app/api/main.py`

- [ ] **Step 1: Aggiornare require_role in deps.py**

In `backend/app/api/deps.py`, sostituire la funzione `require_role` con:

```python
def require_role(min_role: str):
    def _check(current_user: dict = Depends(get_current_user)):
        role = current_user.get("role", "")
        if role == "superadmin":
            return current_user
        if role == min_role:
            return current_user
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return _check
```

- [ ] **Step 2: Aggiungere endpoint DELETE in routes/auth.py**

In `backend/app/api/routes/auth.py`, alla fine del file (dopo `generate_invites_endpoint`), aggiungere:

```python
@router.delete("/auth/users/{username}")
async def delete_user_endpoint(
    username: str,
    current_user: dict = Depends(require_role("admin")),
):
    """Delete a user. Superadmin can delete anyone, admin can only delete users."""
    delete_user(username, current_user["username"])
    return {"deleted": username}
```

Aggiungere l'import mancante in cima al file:
```python
from app.core.auth import delete_user
```

- [ ] **Step 3: Cambiare bootstrap in main.py**

In `backend/app/api/main.py`, trovare la riga:
```python
ensure_admin_exists()
```
e cambiarla in:
```python
ensure_superadmin_exists()
```

Aggiornare anche l'import:
```python
from app.core.auth import ensure_superadmin_exists
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/api/deps.py backend/app/api/routes/auth.py backend/app/api/main.py
git commit -m "feat: DELETE /auth/users/{username}, require_role gerarchia, superadmin bootstrap"
```

---

### Task 4: Frontend — types + api + route + sidebar

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/layout/Sidebar.tsx`

- [ ] **Step 1: Aggiungere deleteUser in types.ts**

In `frontend/src/lib/types.ts`, in cima dopo gli import esistenti, aggiungere il tipo:

```typescript
export type Role = "user" | "admin" | "superadmin";
```

Poi nell'interfaccia `ExperimentSummary` **non** serve cambiare nulla. `Role` è un tipo esportato per uso nei componenti.

- [ ] **Step 2: Aggiungere deleteUser in api.ts**

In `frontend/src/lib/api.ts`, alla fine del file (dopo `fetchMe`), aggiungere:

```typescript
export async function deleteUser(username: string): Promise<void> {
  await fetchApi(`/auth/users/${encodeURIComponent(username)}`, {
    method: "DELETE",
  });
}
```

- [ ] **Step 3: Cambiare route in App.tsx**

In `frontend/src/App.tsx`, trovare la riga:
```tsx
<Route path="/invites" element={<Lazy cmp={() => import("./pages/InvitePage")} />} />
```
e cambiarla in:
```tsx
<Route path="/admin/users" element={<Lazy cmp={() => import("./pages/UserManagementPage")} />} />
```

- [ ] **Step 4: Aggiornare sidebar Sidebar.tsx**

In `frontend/src/components/layout/Sidebar.tsx`:

1. Aggiungere `Users` all'import da `lucide-react` e rimuovere `KeyRound` (non più usato):
```tsx
import {
  LayoutDashboard,
  FlaskConical,
  ScanSearch,
  Box,
  Database,
  BookOpen,
  X,
  Users,
} from "lucide-react";
```

2. Trovare il blocco condizionale esistente per l'admin link (che inizia con `{user?.role === "admin" && (` e contiene `<NavLink to="/invites"`). Sostituire INTERAMENTE quel blocco con:

```tsx
{user?.role === "admin" && (
  <NavLink
    to="/admin/users"
    onClick={() => setSidebarOpen(false)}
    className={({ isActive }) =>
      cn(
        "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors duration-150 cursor-pointer",
        isActive
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:bg-muted hover:text-foreground"
      )
    }
  >
    <Users className="size-5 shrink-0" />
    <span>{t("sidebar.user_management")}</span>
  </NavLink>
)}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/App.tsx frontend/src/components/layout/Sidebar.tsx
git commit -m "feat: deleteUser API, route /admin/users, sidebar User Management"
```

---

### Task 5: Frontend — pagina UserManagementPage

**Files:**
- Rename: `frontend/src/pages/InvitePage.tsx` → `frontend/src/pages/UserManagementPage.tsx`
- Modify: `frontend/src/pages/UserManagementPage.tsx`

- [ ] **Step 1: Rinominare il file**

```bash
mv frontend/src/pages/InvitePage.tsx frontend/src/pages/UserManagementPage.tsx
```

Poi rimuovere il vecchio file dal tracking e aggiungere il nuovo:
```bash
git rm frontend/src/pages/InvitePage.tsx
git add frontend/src/pages/UserManagementPage.tsx
```

- [ ] **Step 2: Modificare UserManagementPage.tsx**

Aprire `frontend/src/pages/UserManagementPage.tsx`.

**Import:** Aggiungere `Dialog`, `DialogContent`, `DialogHeader`, `DialogTitle`, `DialogDescription`, `DialogFooter` da `@/components/ui/dialog`. Importare `deleteUser` da `@/lib/api`.

Aggiungere questi import in cima:
```tsx
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { deleteUser } from "@/lib/api";
```

**State:** Aggiungere dopo gli useState esistenti:
```tsx
const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
```

**Funzione conferma elimina:** Aggiungere prima del JSX di return:
```tsx
async function confirmDelete() {
  if (!deleteTarget) return;
  try {
    await deleteUser(deleteTarget);
    setDeleteTarget(null);
    const data = await fetchInvites();
    setUsers(data.users);
  } catch {
    setDeleteTarget(null);
  }
}
```

**Tabella utenti:** Nella sezione users, dopo la colonna `<TableHead>{t("invites.role")}</TableHead>`, aggiungere:
```tsx
<TableHead className="text-right">{t("admin_users.actions")}</TableHead>
```

E in ogni riga, dopo il `<Badge>`, aggiungere:
```tsx
<TableCell className="text-right">
  {(currentUser.role === "superadmin" || (currentUser.role === "admin" && u.role === "user")) && (
    <Button
      variant="destructive"
      size="sm"
      onClick={() => setDeleteTarget(u.username)}
    >
      {t("admin_users.delete")}
    </Button>
  )}
</TableCell>
```

**Variabile currentUser:** Aggiungere in cima alla funzione del componente (dopo gli hook):
```tsx
const currentUser = useAppStore((s) => ({ username: s.user?.username, role: s.user?.role }));
```

**Confirm dialog:** Aggiungere da qualche parte nel JSX (dopo la chiusura di `</div>`, prima di `</>`):
```tsx
<Dialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>{t("admin_users.delete_confirm_title")}</DialogTitle>
      <DialogDescription>
        {t("admin_users.delete_confirm_body", { username: deleteTarget })}
      </DialogDescription>
    </DialogHeader>
    <DialogFooter>
      <Button variant="outline" onClick={() => setDeleteTarget(null)}>
        {t("common.cancel")}
      </Button>
      <Button variant="destructive" onClick={confirmDelete}>
        {t("admin_users.delete")}
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
```

- [ ] **Step 3: Verificare build TypeScript**

```bash
cd frontend && npx tsc --noEmit
```
Nessun errore.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/UserManagementPage.tsx
git commit -m "feat: UserManagementPage con delete utenti e confirm dialog"
```

---

### Task 6: Frontend — i18n

**Files:**
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/it.json`

- [ ] **Step 1: Aggiungere chiavi en.json**

In `frontend/src/locales/en.json`:

1. Nel blocco `"sidebar"`, aggiungere:
```json
"user_management": "User Management"
```

2. Aggiungere un NUOVO blocco `"admin_users"` (da qualche parte, ad esempio dopo `"invites"`):
```json
"admin_users": {
  "title": "User Management",
  "username": "Username",
  "role": "Role",
  "actions": "Actions",
  "delete": "Delete",
  "delete_confirm_title": "Delete user",
  "delete_confirm_body": "Are you sure you want to delete {{username}}? This action cannot be undone.",
  "cant_delete_self": "You cannot delete your own account"
}
```

- [ ] **Step 2: Aggiungere chiavi it.json**

In `frontend/src/locales/it.json`:

1. Nel blocco `"sidebar"`, aggiungere:
```json
"user_management": "Gestione Utenti"
```

2. Blocco `"admin_users"`:
```json
"admin_users": {
  "title": "Gestione Utenti",
  "username": "Utente",
  "role": "Ruolo",
  "actions": "Azioni",
  "delete": "Elimina",
  "delete_confirm_title": "Elimina utente",
  "delete_confirm_body": "Sei sicuro di voler eliminare {{username}}? Questa azione non può essere annullata.",
  "cant_delete_self": "Non puoi eliminare il tuo account"
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/locales/en.json frontend/src/locales/it.json
git commit -m "feat: i18n per admin_users e sidebar.user_management"
```

---

### Task 7: Verifica build finale

- [ ] **Step 1: Build frontend**

```bash
cd frontend && npm run build
```

Nessun errore TypeScript o Vite.

- [ ] **Step 2: Verifica backend**

```bash
cd backend && python3 -c "from app.core.auth import delete_user, ensure_superadmin_exists; print('OK')"
```

- [ ] **Step 3: Aggiornare .env sulla VPS**

L'`.env` reale sulla VPS (ignorato da git) ha ancora `ADMIN_USERNAME/PASSWORD`. Dopo il deploy:

```bash
ssh -i keys/orchid-osr ubuntu@51.38.51.220 "sed -i 's/ADMIN_USERNAME/SUPERADMIN_USERNAME/g; s/ADMIN_PASSWORD/SUPERADMIN_PASSWORD/g' ~/OrchidOSR/.env"
```

Poi dopo il prossimo deploy automatico, il container userà le nuove var.

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

### Riepilogo file per task

| Task | File | Modifica |
|------|------|----------|
| 1 | `backend/app/config.py` | `admin` → `superadmin` |
| 1 | `.env.example` | `SUPERADMIN_USERNAME/PASSWORD` |
| 1 | `docker-compose.yml` | env var rename |
| 1 | `entrypoint.sh` | env var rename |
| 2 | `backend/app/core/auth.py` | `ensure_superadmin_exists`, `delete_user`, `ROLE_HIERARCHY` |
| 3 | `backend/app/api/deps.py` | `require_role` lascia passare superadmin |
| 3 | `backend/app/api/routes/auth.py` | `DELETE /auth/users/{username}` |
| 3 | `backend/app/api/main.py` | bootstrap rename |
| 4 | `frontend/src/lib/types.ts` | `Role` type |
| 4 | `frontend/src/lib/api.ts` | `deleteUser()` |
| 4 | `frontend/src/App.tsx` | route `/invites` → `/admin/users` |
| 4 | `frontend/src/components/layout/Sidebar.tsx` | rename + icona `Users` |
| 5 | `frontend/src/pages/InvitePage.tsx` → `UserManagementPage.tsx` | rename + delete |
| 6 | `frontend/src/locales/en.json` | nuove chiavi admin_users |
| 6 | `frontend/src/locales/it.json` | nuove chiavi admin_users |
