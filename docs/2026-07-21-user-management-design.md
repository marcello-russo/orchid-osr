# User Management Page — Design Doc

## Objective

Replace the basic Invites page with a full User Management panel at `/admin/users`. Introduce a `superadmin` role bootstrapped from `.env`, add user deletion with role-based permissions, and rename the navigation.

## Role Hierarchy

```
superadmin  (bootstrap da .env)
    ↓
   admin    (creato via inviti con role=admin)
    ↓
   user     (creato via inviti con role=user)
```

Regole delete:
- Un utente NON può eliminare sé stesso
- Superadmin può eliminare qualsiasi ruolo (user, admin, altro superadmin)
- Admin può eliminare solo user (non admin, non superadmin)
- User non può eliminare nessuno (non ha accesso alla pagina)

## Backend Changes

### 1. `backend/app/config.py`

**Rimuovere:**
```python
admin_username: str = ""
admin_password: str = ""
```

**Aggiungere:**
```python
superadmin_username: str = ""
superadmin_password: str = ""
```

Rimuovere anche il model_validator `_validate_admin_credentials` (non più necessario).

### 2. `backend/app/core/auth.py`

**2a. `ensure_admin_exists()` → `ensure_superadmin_exists()`**

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

**2b. Nuova funzione `delete_user()`**

```python
ROLE_HIERARCHY = {"user": 0, "admin": 1, "superadmin": 2}

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
        # Reset invites used by this user
        for invite in data["invites"]:
            if invite.get("used_by") == target_username:
                invite["used"] = False
                invite.pop("used_by", None)

    _atomic_modify(_modify)
```

**2c. `list_users_and_invites()`** — già ok, ritorna `{"users": [...], "invites": [...]}` con i ruoli.

### 3. `backend/app/api/routes/auth.py`

**3a. Aggiungere endpoint `DELETE /auth/users/{username}`**

```python
@router.delete("/auth/users/{username}")
async def delete_user_endpoint(
    username: str,
    current_user: dict = Depends(require_role("admin")),
):
    """Delete a user. Requires admin+; superadmin can delete anyone, admin only users."""
    auth_module.delete_user(username, current_user["username"])
    return {"deleted": username}
```

**3b. Cambiare bootstrap da admin a superadmin**

Nel file `__init__.py` o in `server.py` o in `main.py` dove viene chiamato `ensure_admin_exists()`, cambiare con `ensure_superadmin_exists()`.

### 4. `backend/app/api/deps.py`

`require_role("admin")` — già funziona: superadmin ha ruolo `superadmin` che è diverso da `admin`. Bisogna aggiornare la logica perché anche `superadmin` possa passare i controlli `admin`.

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

### 5. `backend/.env.example`

```
SUPERADMIN_USERNAME=admin
SUPERADMIN_PASSWORD=admin
```

### 6. `docker-compose.yml` e `entrypoint.sh`

Cambiare i riferimenti da `ADMIN_USERNAME`/`ADMIN_PASSWORD` a `SUPERADMIN_USERNAME`/`SUPERADMIN_PASSWORD`.

## Frontend Changes

### 7. Route: `/invites` → `/admin/users`

**`frontend/src/App.tsx`:**
```tsx
<Route element={<ProtectedRoute role="admin" />}>
  <Route path="/dataset" element={<Lazy cmp={() => import("./pages/DatasetPage")} />} />
  <Route path="/admin/users" element={<Lazy cmp={() => import("./pages/UserManagementPage")} />} />
</Route>
```

### 8. Pagina: `InvitePage.tsx` → `UserManagementPage.tsx`

Rinominare il file. Stessa struttura ma con aggiunte:

**Sezione utenti** — aggiungere colonna "Actions":

```tsx
<Table>
  <thead>
    <tr>
      <th>{t("admin_users.username")}</th>
      <th>{t("admin_users.role")}</th>
      <th>{t("admin_users.actions")}</th>
    </tr>
  </thead>
  <tbody>
    {users.map((u) => (
      <tr key={u.username}>
        <td>{u.username}</td>
        <td><RoleBadge role={u.role} /></td>
        <td>
          {canDelete(u.role, currentUser.role) && (
            <Button variant="destructive" size="sm" onClick={() => handleDelete(u.username)}>
              {t("admin_users.delete")}
            </Button>
          )}
        </td>
      </tr>
    ))}
  </tbody>
</Table>
```

**Logica `canDelete`:**
```typescript
function canDelete(targetRole: string, myRole: string): boolean {
  if (myRole === "superadmin") return true;
  if (myRole === "admin" && targetRole === "user") return true;
  return false;
}
```

**Confirm dialog** — prima di eliminare:

```tsx
const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

// ... nel JSX:
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

**Funzione API `deleteUser`:**
```typescript
export async function deleteUser(username: string): Promise<void> {
  const res = await apiFetch(`/auth/users/${encodeURIComponent(username)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
}
```

Nota: l'API `fetchInvites()` già ritorna sia invites che users. Dopo delete, rifare `fetchInvites()` per aggiornare la lista.

### 9. Sidebar: rename + icon

**`frontend/src/components/layout/Sidebar.tsx`:**

```tsx
import { Users } from "lucide-react";

// Nel navItems o condizione admin:
{user?.role === "admin" && (
  <NavLink
    to="/admin/users"
    onClick={() => setSidebarOpen(false)}
    className={({ isActive }) => cn(...)}
  >
    <Users className="size-5 shrink-0" />
    <span>{t("sidebar.user_management")}</span>
  </NavLink>
)}
```

### 10. i18n

**`en.json`:**
```json
"sidebar": {
  "user_management": "User Management"
},
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

**`it.json`:**
```json
"sidebar": {
  "user_management": "Gestione Utenti"
},
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

## Files modificati

| File | Modifica |
|------|----------|
| `backend/app/config.py` | `admin_username/password` → `superadmin_username/password` |
| `backend/app/core/auth.py` | `ensure_superadmin_exists()`, `delete_user()`, gerarchia ruoli |
| `backend/app/api/routes/auth.py` | Nuovo endpoint `DELETE /auth/users/{username}` |
| `backend/app/api/main.py` (o server.py) | `ensure_admin_exists()` → `ensure_superadmin_exists()` |
| `backend/app/api/deps.py` | `require_role()` lascia passare superadmin |
| `.env.example` | `SUPERADMIN_USERNAME/PASSWORD` |
| `docker-compose.yml` | Variabili d'ambiente aggiornate |
| `entrypoint.sh` | Variabili d'ambiente aggiornate |
| `frontend/src/pages/InvitePage.tsx` | Rinomina in `UserManagementPage.tsx` + delete utenti |
| `frontend/src/App.tsx` | Route `/invites` → `/admin/users` |
| `frontend/src/components/layout/Sidebar.tsx` | Rename + icona Users |
| `frontend/src/lib/api.ts` | Aggiungere `deleteUser()` |
| `frontend/src/lib/types.ts` | Eventuale tipo `Role = "user" \| "admin" \| "superadmin"` |
| `frontend/src/locales/en.json` | Nuove chiavi |
| `frontend/src/locales/it.json` | Nuove chiavi |

## Test

- Superadmin: eliminare user → OK
- Superadmin: eliminare admin → OK
- Admin: eliminare user → OK
- Admin: eliminare admin → 403
- Admin: eliminare superadmin → 403
- Chiunque: eliminare sé stesso → 400
- Verificare che dopo delete l'utente non possa più fare login
- Verificare che gli inviti dell'utente eliminato vengano resettati
