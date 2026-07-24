# Backend API Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Portare la copertura backend dal 40% al 44% testando tutti gli endpoint API mancanti (auth, admin, dataset, experiment detail, embeddings).

**Architecture:** 5 nuovi file test in `backend/tests/test_api/`, ognuno con TestClient FastAPI. Fixture condivise per auth e filesystem finto. Ogni endpoint testato: happy path + errori + validazione.

**Tech Stack:** pytest + FastAPI TestClient + monkeypatch + tmp_path

---

## File Structure

```
backend/tests/
├── conftest.py                          # Fixture condivise (già esistenti)
├── test_api/
│   ├── __init__.py                      # Già esistente
│   ├── conftest.py                      # [DA MODIFICARE] Fixture specifiche API
│   ├── test_health.py                   # Già esistente
│   ├── test_classify.py                 # Già esistente
│   ├── test_classify_upload.py          # Già esistente
│   ├── test_experiments_list.py         # Già esistente
│   ├── test_species.py                  # Già esistente
│   ├── test_auth.py                     # [NUOVO]   Register, login, me, refresh
│   ├── test_admin.py                    # [NUOVO]   Delete user, list users
│   ├── test_dataset.py                  # [NUOVO]   Dataset summary, gallery
│   ├── test_experiments_detail.py       # [NUOVO]   GET /api/experiments/{id}
│   └── test_embeddings.py              # [NUOVO]   GET /api/embeddings
```

---

### Task 0: Auth fixture — database temporaneo + superadmin + invite codes

**Attenzione:** I test auth/admin interagiscono con un database SQLite JSON. Per isolarli:
- Ogni test usa un database temporaneo (`tmp_path`)
- `monkeypatch` reindirizza `app.core.auth._get_data_path()` al db temporaneo
- `ensure_superadmin_exists()` viene chiamato dopo aver impostato le env var
- `generate_invite()` crea invite codes validi prima dei test di registrazione

**Files:**
- Modify: `backend/tests/test_api/conftest.py`

- [ ] **Step 0: Scrivere fixture condivise per auth**

```python
# backend/tests/test_api/conftest.py — AGGIUNGERE in fondo
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from app.api.main import app


@pytest.fixture
def client():
    """TestClient senza autenticazione."""
    return TestClient(app)


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    """Crea database auth temporaneo con superadmin e invite codes."""
    from app.core import auth as auth_module

    # Crea database vuoto
    db_path = tmp_path / "auth_db.json"
    db_path.write_text(json.dumps({"invites": [], "users": []}))
    monkeypatch.setattr(auth_module, "_get_data_path", lambda: db_path)

    # Bootstrap superadmin da env var
    monkeypatch.setenv("SUPERADMIN_USERNAME", "testsuperadmin")
    monkeypatch.setenv("SUPERADMIN_PASSWORD", "TestAdmin2026!")
    auth_module.ensure_superadmin_exists()

    # Crea invite codes per i test
    auth_module.generate_invite(role="user", count=5)
    auth_module.generate_invite(role="admin", count=2)

    return db_path


@pytest.fixture
def superadmin_token(auth_db, client):
    """Token di autenticazione come superadmin."""
    resp = client.post("/auth/login", json={
        "username": "testsuperadmin",
        "password": "TestAdmin2026!",
    })
    return resp.json()["token"]


@pytest.fixture
def user_token(auth_db, client):
    """Registra e logga un utente normale, restituisce il token."""
    client.post("/auth/register", json={
        "username": "testuser",
        "password": "Password1",
        "invite_code": "invite_0",
    })
    resp = client.post("/auth/login", json={
        "username": "testuser",
        "password": "Password1",
    })
    return resp.json()["token"]
```

---

### Task 1: Auth API tests — register, login, me

**Files:**
- Create: `backend/tests/test_api/test_auth.py`

**Dipende da:** Task 0 (fixture auth_db, superadmin_token, user_token)

- [ ] **Step 1: Scrivere test per registrazione**

```python
# backend/tests/test_api/test_auth.py

class TestRegister:
    """POST /auth/register — usa fixture auth_db per database temporaneo."""

    def test_register_success(self, client, auth_db):
        """Registrazione con invite code valido."""
        response = client.post("/auth/register", json={
            "username": "newuser",
            "password": "Password1",
            "invite_code": "invite_0",
        })
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert data["username"] == "newuser"
        assert data["role"] == "user"

    def test_register_duplicate_username(self, client, auth_db):
        """Stesso username due volte."""
        client.post("/auth/register", json={
            "username": "dupuser", "password": "Password1", "invite_code": "invite_1",
        })
        response = client.post("/auth/register", json={
            "username": "dupuser", "password": "Password2", "invite_code": "invite_2",
        })
        assert response.status_code == 400

    def test_register_weak_password(self, client, auth_db):
        """Password senza maiuscola."""
        response = client.post("/auth/register", json={
            "username": "weakuser", "password": "onlylowercase", "invite_code": "invite_3",
        })
        assert response.status_code == 422

    def test_register_invalid_invite(self, client, auth_db):
        """Codice invito inesistente."""
        response = client.post("/auth/register", json={
            "username": "badinvite", "password": "Password1", "invite_code": "XXXXX",
        })
        assert response.status_code == 400
```

- [ ] **Step 2: Scrivere test per login**

```python
class TestLogin:
    """POST /auth/login."""

    def test_login_success(self, client, auth_db):
        """Login con credenziali valide."""
        client.post("/auth/register", json={
            "username": "loginuser", "password": "Password1", "invite_code": "invite_4",
        })
        response = client.post("/auth/login", json={
            "username": "loginuser", "password": "Password1",
        })
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert data["username"] == "loginuser"

    def test_login_wrong_password(self, client, auth_db):
        """Password errata."""
        response = client.post("/auth/login", json={
            "username": "testsuperadmin", "password": "WRONG",
        })
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client, auth_db):
        """Utente inesistente."""
        response = client.post("/auth/login", json={
            "username": "ghost", "password": "Password1",
        })
        assert response.status_code == 401
```

- [ ] **Step 3: Scrivere test per me**

```python
class TestMe:
    """GET /auth/me."""

    def test_me_authenticated(self, client, user_token):
        """Utente autenticato."""
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {user_token}"})
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert "role" in data

    def test_me_unauthenticated(self, client):
        """Senza token."""
        response = client.get("/auth/me")
        assert response.status_code == 401
```

- [ ] **Step 4: Eseguire i test**

```bash
cd backend && python3 -m pytest tests/test_api/test_auth.py -v
```
Expected: 9 test passano

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_api/conftest.py backend/tests/test_api/test_auth.py
git commit -m "test: auth API — register, login, me (9 test)"
```

---

### Task 2: Admin API tests — delete user, list users

**Files:**
- Create: `backend/tests/test_api/test_admin.py`

**Dipende da:** Task 0 (fixture auth_db, superadmin_token, user_token)

- [ ] **Step 1: Scrivere test per admin delete user**

```python
# backend/tests/test_api/test_admin.py

class TestAdminDeleteUser:
    """DELETE /auth/users/{username}"""

    def test_superadmin_can_delete_user(self, client, auth_db, superadmin_token):
        """Superadmin elimina un utente normale."""
        client.post("/auth/register", json={
            "username": "todelete", "password": "Password1", "invite_code": "invite_0",
        })
        response = client.delete(
            "/auth/users/todelete",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response.status_code == 200

    def test_delete_nonexistent_user(self, client, auth_db, superadmin_token):
        """Elimina utente inesistente."""
        response = client.delete(
            "/auth/users/nobody",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response.status_code == 404

    def test_delete_unauthorized(self, client):
        """Senza token."""
        response = client.delete("/auth/users/someuser")
        assert response.status_code == 401

    def test_user_cannot_delete_other(self, client, auth_db, user_token):
        """Utente normale non può eliminare altri."""
        response = client.delete(
            "/auth/users/testsuperadmin",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 403
```

- [ ] **Step 2: Scrivere test per list users**

```python
class TestAdminListUsers:
    """GET /auth/users"""

    def test_list_users_as_superadmin(self, client, auth_db, superadmin_token):
        """Superadmin vede tutti gli utenti."""
        response = client.get("/auth/users", headers={"Authorization": f"Bearer {superadmin_token}"})
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert len(data["users"]) >= 1

    def test_list_users_as_user_denied(self, client, auth_db, user_token):
        """Utente normale non può listare."""
        response = client.get("/auth/users", headers={"Authorization": f"Bearer {user_token}"})
        assert response.status_code == 403

    def test_list_users_unauthorized(self, client):
        """Senza token."""
        response = client.get("/auth/users")
        assert response.status_code == 401
```

- [ ] **Step 3: Eseguire i test**

```bash
cd backend && python3 -m pytest tests/test_api/test_admin.py -v
```
Expected: 7 test passano

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_api/test_admin.py
git commit -m "test: admin API — delete user, list users (7 test)"
```

---

### Task 3: Dataset API tests

**Files:**
- Create: `backend/tests/test_api/test_dataset.py`

- [ ] **Step 1: Scrivere test per dataset summary**

```python
# backend/tests/test_api/test_dataset.py
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


class TestDatasetSummary:
    """GET /api/dataset/summary"""

    def _setup_fake_dataset(self, tmp_path: Path):
        """Crea struttura dataset finto per i test."""
        split_dirs = ["train", "test"]
        species = ["Ophrys_exaltata", "Ophrys_garganica"]
        for split in split_dirs:
            for sp in species:
                sp_dir = tmp_path / split / sp
                sp_dir.mkdir(parents=True)
                for i in range(3):
                    (sp_dir / f"img_{i}.jpg").write_text("fake image data")
        return tmp_path

    def test_summary_returns_data(self, tmp_path, monkeypatch):
        """Dataset summary con dati reali."""
        fake = self._setup_fake_dataset(tmp_path)
        monkeypatch.setattr("app.api.routes.dataset._get_dataset_dirs", lambda: [fake])

        response = client.get("/api/dataset/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_species" in data
        assert data["total_species"] == 2
        assert "total_images" in data
        assert data["total_images"] == 12  # 2 species × 2 split × 3 images

    def test_summary_empty(self, tmp_path, monkeypatch):
        """Dataset summary senza dati."""
        monkeypatch.setattr("app.api.routes.dataset._get_dataset_dirs", lambda: [tmp_path])
        response = client.get("/api/dataset/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_species"] == 0
```

- [ ] **Step 2: Scrivere test per dataset gallery**

```python
class TestDatasetGallery:
    """GET /api/dataset/gallery"""

    def _setup_fake_dataset(self, tmp_path: Path):
        split_dirs = ["train", "test"]
        species = ["Ophrys_exaltata"]
        for split in split_dirs:
            for sp in species:
                sp_dir = tmp_path / split / sp
                sp_dir.mkdir(parents=True)
                for i in range(2):
                    (sp_dir / f"img_{i}.jpg").write_text("fake")
        return tmp_path

    def test_gallery_returns_images(self, tmp_path, monkeypatch):
        """Gallery restituisce path immagini."""
        fake = self._setup_fake_dataset(tmp_path)
        monkeypatch.setattr("app.api.routes.dataset._get_dataset_dirs", lambda: [fake])

        response = client.get("/api/dataset/gallery")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
```

- [ ] **Step 3: Eseguire i test**

Run: `cd backend && python3 -m pytest tests/test_api/test_dataset.py -v`
Expected: 3 test passano

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_api/test_dataset.py
git commit -m "test: dataset API — summary, gallery (3 test)"
```

---

### Task 4: Experiment detail API tests

**Files:**
- Create: `backend/tests/test_api/test_experiments_detail.py`

- [ ] **Step 1: Scrivere test helper per creare esperimenti finti**

```python
# backend/tests/test_api/test_experiments_detail.py
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def _make_exp(exp_dir: Path, name: str, config: dict, results: dict, folds: int = 3):
    """Crea un esperimento finto con config, results e fold file."""
    d = exp_dir / name
    d.mkdir(parents=True)
    with open(d / "config.json", "w") as f:
        json.dump({"name": name, "model_variant": "resnet18", **config}, f)
    with open(d / "results.json", "w") as f:
        json.dump({"mean_f1": 0.85, "std_f1": 0.02, **results}, f)
    for i in range(folds):
        fold_data = {
            "best_f1": 0.84 + i * 0.01,
            "history": {"train_loss": [1.0, 0.5], "val_loss": [0.8, 0.4]},
        }
        with open(d / f"fold_{i}.json", "w") as f:
            json.dump(fold_data, f)
```

- [ ] **Step 2: Scrivere test per experiment detail**

```python
class TestExperimentDetail:
    """GET /api/experiments/{id}"""

    def test_detail_existing(self, tmp_path, monkeypatch):
        """Dettaglio esperimento esistente."""
        from app.api.routes import experiments as exp_route
        monkeypatch.setattr(exp_route, "_get_experiments_dir", lambda: tmp_path)
        _make_exp(tmp_path, "exp6_test", {"num_folds": 3}, {"test_metrics": {"accuracy": 0.82}}, folds=3)

        response = client.get("/api/experiments/exp6_test")
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["id"] == "exp6_test"
        assert "config" in data
        assert "folds" in data
        assert len(data["folds"]) == 3

    def test_detail_not_found(self, tmp_path, monkeypatch):
        """Dettaglio esperimento inesistente."""
        from app.api.routes import experiments as exp_route
        monkeypatch.setattr(exp_route, "_get_experiments_dir", lambda: tmp_path)

        response = client.get("/api/experiments/nonexistent_exp")
        assert response.status_code == 404

    def test_detail_without_folds(self, tmp_path, monkeypatch):
        """Dettaglio esperimento senza fold."""
        from app.api.routes import experiments as exp_route
        monkeypatch.setattr(exp_route, "_get_experiments_dir", lambda: tmp_path)
        _make_exp(tmp_path, "baseline_test", {"num_folds": 0}, {"test_metrics": {"accuracy": 0.90}}, folds=0)

        response = client.get("/api/experiments/baseline_test")
        assert response.status_code == 200
        data = response.json()
        assert len(data["folds"]) == 0
```

- [ ] **Step 3: Eseguire i test**

Run: `cd backend && python3 -m pytest tests/test_api/test_experiments_detail.py -v`
Expected: 3 test passano

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_api/test_experiments_detail.py
git commit -m "test: experiment detail API — dettaglio, 404, senza fold (3 test)"
```

---

### Task 5: Embeddings API tests

**Files:**
- Create: `backend/tests/test_api/test_embeddings.py`

- [ ] **Step 1: Scrivere test per embeddings list**

```python
# backend/tests/test_api/test_embeddings.py
import json
import numpy as np
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def _make_embeddings(exp_dir: Path, exp_name: str, n_samples: int = 10):
    """Crea file embedding finti per un esperimento."""
    d = exp_dir / exp_name / "embeddings"
    d.mkdir(parents=True)
    np.save(d / "embeddings.npy", np.random.randn(n_samples, 128).astype(np.float32))
    np.save(d / "labels.npy", np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9][:n_samples]))
    with open(d / "metadata.json", "w") as f:
        json.dump({
            "experiment_id": exp_name,
            "model_variant": "resnet18",
            "num_samples": n_samples,
            "embedding_dim": 128,
        }, f)


class TestEmbeddings:
    """GET /api/embeddings"""

    def test_list_embeddings_with_data(self, tmp_path, monkeypatch):
        """Lista embedding con dati."""
        from app.api.routes import embeddings as emb_route
        monkeypatch.setattr(emb_route, "_get_experiments_dir", lambda: tmp_path)
        _make_embeddings(tmp_path, "exp8_emb_test")

        response = client.get("/api/embeddings")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_list_embeddings_empty(self, tmp_path, monkeypatch):
        """Lista embedding senza dati."""
        from app.api.routes import embeddings as emb_route
        monkeypatch.setattr(emb_route, "_get_experiments_dir", lambda: tmp_path)

        response = client.get("/api/embeddings")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_embeddings_filter_by_experiment(self, tmp_path, monkeypatch):
        """Filtra embedding per experiment ID."""
        from app.api.routes import embeddings as emb_route
        monkeypatch.setattr(emb_route, "_get_experiments_dir", lambda: tmp_path)
        _make_embeddings(tmp_path, "exp8_target")
        _make_embeddings(tmp_path, "exp8_other")

        response = client.get("/api/embeddings?experiment_id=exp8_target")
        assert response.status_code == 200
        data = response.json()
        # Tutti gli elementi devono avere experiment_id = "exp8_target"
        for item in data:
            assert item.get("experiment_id") == "exp8_target"
```

- [ ] **Step 2: Eseguire i test**

Run: `cd backend && python3 -m pytest tests/test_api/test_embeddings.py -v`
Expected: 3 test passano

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_api/test_embeddings.py
git commit -m "test: embeddings API — list, empty, filter (3 test)"
```

---

### Task 6: Verifica coverage

- [ ] **Step 1: Eseguire tutti i test API con coverage**

```bash
cd backend
python3 -m pytest tests/test_api/ -v --cov=app --cov=scripts --cov-report=term --cov-fail-under=0
```

Expected: ~22 test passano (8 auth + 5 admin + 3 dataset + 3 detail + 3 embeddings + pre-esistenti)

- [ ] **Step 2: Verificare incremento coverage**

La copertura `app/` dovrebbe aumentare di ~2-3%.

- [ ] **Step 3: Eseguire l'intera suite backend**

```bash
cd backend
python3 -m pytest -x -q --cov=app --cov=scripts --cov-report=term --cov-fail-under=70
```

Expected: 70% di copertura (o il nuovo valore dopo l'incremento)

- [ ] **Step 4: Commit finale**

```bash
git add -A
git commit -m "test: backend API — copertura completa auth/admin/dataset/detail/embeddings"
```

---

## Riepilogo

| Task | File | Test |
|------|------|------|
| 1 | `test_auth.py` | 8 test: register (4), login (3), me (2) |
| 2 | `test_admin.py` | 5 test: delete user (3), list users (2) |
| 3 | `test_dataset.py` | 3 test: summary (2), gallery (1) |
| 4 | `test_experiments_detail.py` | 3 test: detail (1), 404 (1), senza fold (1) |
| 5 | `test_embeddings.py` | 3 test: list (1), empty (1), filter (1) |
| **Totale** | **5 nuovi file** | **~22 test** |
