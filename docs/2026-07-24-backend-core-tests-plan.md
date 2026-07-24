# Backend Core Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Portare la copertura backend dal ~44% al 65% testando moduli core (auth, kfold, supcon, preprocessing).

**Architecture:** File test in `backend/tests/test_core/` e `backend/tests/test_preprocessing/`. Ogni modulo testato in isolamento con mock di dipendenze pesanti (torch, filesystem).

**Tech Stack:** pytest + torch + unittest.mock + tmp_path

---

## File Structure

```
backend/tests/
├── test_core/
│   ├── test_auth.py              # [NUOVO]   Password hashing, JWT, role hierarchy
│   ├── test_kfold_training.py    # [NUOVO]   Training loop, checkpoint, fold resume
│   └── test_supcon.py            # [NUOVO]   SupConLoss forward/backward
├── test_preprocessing/
│   ├── test_split.py             # [NUOVO]   Stratified split, MD5 dedup
│   └── test_cropper.py           # [NUOVO]   Crop, no detection fallback
```

---

### Task 1: Core auth tests — password, JWT, roles

**Files:**
- Create: `backend/tests/test_core/test_auth.py`

- [ ] **Step 1: Test password hashing e verifica**

```python
from app.core.auth import hash_password, verify_password

class TestPassword:
    def test_hash_and_verify(self):
        h = hash_password("Password1!")
        assert verify_password("Password1!", h)
    
    def test_wrong_password_fails(self):
        h = hash_password("Password1!")
        assert not verify_password("WrongPass1!", h)
    
    def test_different_hash_each_time(self):
        h1 = hash_password("Password1!")
        h2 = hash_password("Password1!")
        assert h1 != h2  # salt diverso
```

- [ ] **Step 2: Test JWT encode/decode**

```python
from app.core.auth import create_jwt, decode_jwt

class TestJWT:
    def test_create_and_decode(self):
        token = create_jwt("testuser", "admin")
        payload = decode_jwt(token)
        assert payload["username"] == "testuser"
        assert payload["role"] == "admin"
    
    def test_expired_token_raises(self):
        # Crea token già scaduto
        pass  # da implementare
```

- [ ] **Step 3: Test role hierarchy**

```python
from app.core.auth import ROLE_HIERARCHY, require_role

class TestRoles:
    def test_hierarchy_order(self):
        assert ROLE_HIERARCHY["user"] < ROLE_HIERARCHY["admin"] < ROLE_HIERARCHY["superadmin"]
```

- [ ] **Step 4: Eseguire e fixare**

```bash
cd backend && python3 -m pytest tests/test_core/test_auth.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_core/test_auth.py
git commit -m "test: core auth — password, JWT, role hierarchy"
```

---

### Task 2: K-fold training tests

**Files:**
- Create: `backend/tests/test_core/test_kfold_training.py`

Testa il modulo `app/core/kfold/training.py`:
- `train_fold()` con dataloader finto
- `_train_epoch()` con mock del modello
- Checkpoint save/load
- Fold completion logic

- [ ] **Step 1: Test training loop base**

```python
import torch
from app.core.kfold.training import train_fold

class TestTrainFold:
    def test_train_fold_completes(self, tiny_dataloader, device):
        model = torch.nn.Linear(224*224*3, 6).to(device)
        result = train_fold(model, tiny_dataloader, tiny_dataloader, device, num_epochs=2)
        assert result is not None
        assert "best_f1" in result
```

- [ ] **Step 2: Eseguire e fixare**

```bash
cd backend && python3 -m pytest tests/test_core/test_kfold_training.py -v -x
```

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_core/test_kfold_training.py
git commit -m "test: kfold training — train_fold, checkpoint"
```

---

### Task 3: SupCon loss tests

**Files:**
- Create: `backend/tests/test_core/test_supcon.py`

Testa `app/core/losses.py:SupConLoss`:
- Forward pass con features fittizie
- Loss computation for batch con classi multiple
- Edge case: batch con una sola classe
- Temperature scaling

- [ ] **Step 1: Test SupConLoss**

```python
import torch
from app.core.losses import SupConLoss

class TestSupConLoss:
    def test_forward_returns_scalar(self):
        loss_fn = SupConLoss(temperature=0.07)
        features = torch.randn(8, 128)  # batch=8, dim=128
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(features, labels)
        assert loss.ndim == 0  # scalar
        assert loss > 0
    
    def test_same_class_loss(self):
        """Tutti stessa classe → loss più alta (difficile separare)."""
        loss_fn = SupConLoss(temperature=0.07)
        features = torch.randn(8, 128)
        labels = torch.zeros(8, dtype=torch.long)
        loss = loss_fn(features, labels)
        assert loss > 0
```

- [ ] **Step 2: Eseguire e fixare**

```bash
cd backend && python3 -m pytest tests/test_core/test_supcon.py -v
```

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_core/test_supcon.py
git commit -m "test: SupConLoss — forward, same class"
```

---

### Task 4: Preprocessing split tests

**Files:**
- Create: `backend/tests/test_preprocessing/test_split.py`

Testa `app/preprocessing/split.py`:
- Stratified train/test split
- MD5 deduplicazione
- Hold-out extraction
- Seed riproducibilità

- [ ] **Step 1: Test split stratificato**

```python
from pathlib import Path
from app.preprocessing.split import create_stratified_split

class TestSplit:
    def test_stratified_preserves_class_ratio(self, tmp_path):
        # Crea immagini fitte con struttura classi
        for cls in ["cls_a", "cls_b"]:
            (tmp_path / cls).mkdir()
            for i in range(10):
                (tmp_path / cls / f"img_{i}.jpg").write_text("data")
        
        train, test = create_stratified_split(tmp_path, test_ratio=0.2, seed=57)
        assert len(train) > 0
        assert len(test) > 0
        # Verifica proporzione
        assert abs(len(train) / (len(train) + len(test)) - 0.8) < 0.1
```

- [ ] **Step 2: Test MD5 dedup**

```python
class TestMD5:
    def test_dedup_removes_duplicates(self, tmp_path):
        from app.preprocessing.split import md5_dedup
        # Crea immagini identiche (stesso contenuto)
        img = b"same_content"
        for i in range(3):
            (tmp_path / f"img_{i}.jpg").write_bytes(img)
        result = md5_dedup(tmp_path)
        assert len(result) == 1  # solo 1 immagine unica
```

- [ ] **Step 3: Eseguire e fixare**

```bash
cd backend && python3 -m pytest tests/test_preprocessing/test_split.py -v
```

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_preprocessing/test_split.py
git commit -m "test: preprocessing split — stratificato, MD5 dedup"
```

---

### Task 5: Preprocessing cropper tests

**Files:**
- Create: `backend/tests/test_preprocessing/test_cropper.py`

Testa `app/preprocessing/cropper.py`:
- Crop con Faster R-CNN
- Fallback quando nessuna detection
- Immagine vuota / malformata

- [ ] **Step 1: Test crop fallback**

```python
from app.preprocessing.cropper import crop_image

class TestCropper:
    def test_crop_no_detection_returns_original(self, tmp_path):
        """Nessuna detection → immagine originale."""
        from PIL import Image
        img = Image.new("RGB", (100, 100))
        img_path = str(tmp_path / "test.jpg")
        img.save(img_path)
        result = crop_image(img_path)
        assert result is not None
```

- [ ] **Step 2: Eseguire e fixare**

```bash
cd backend && python3 -m pytest tests/test_preprocessing/test_cropper.py -v
```

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_preprocessing/test_cropper.py
git commit -m "test: preprocessing cropper — fallback no detection"
```

---

### Task 6: Verifica coverage

- [ ] **Step 1: Eseguire tutti i test backend**

```bash
cd backend && python3 -m pytest -x -q --cov=app --cov=scripts --cov-report=term --cov-fail-under=60
```

Expected: ~60% di copertura backend.

- [ ] **Step 2: Commit finale**

```bash
git add -A
git commit -m "test: Phase 2 — backend core auth/kfold/supcon/preprocessing"
```
