# Test Suite — Design Doc

> Engineering spec per portare frontend e backend al 70% di copertura con test che catturano bug reali.
> Data: 2026-07-23 | Priorità: Alta

---

## 0. Prerequisito

Prima di iniziare: mergiare PR #89 (tooltip portal + citation popover fix).

---

## 1. Stato attuale

| Area | Statement | Coverage | Target |
|------|-----------|----------|--------|
| Backend `app/` | 5386 | ~60% | 78% |
| Backend `scripts/` | 1270 | ~5% | 30% |
| Backend **totale** | 6657 | **40%** | **70%** |
| Frontend | 1704 | **1.11%** | **70%** |

### Backend: moduli sotto 70%

```
app/cli/experiments.py          19%     CLI commands
app/core/auth.py                56%     Autenticazione JWT
app/core/kfold/training.py      19%     K-fold training loop
app/core/kfold/testing.py       50%     K-fold testing
app/core/kfold/__init__.py      11%     Orchestrazione fold
app/core/losses.py              64%     Funzioni loss
app/core/models/convnext.py     32%     Modello ConvNeXt
app/core/models/dinov2.py       26%     Modello DINOv2
app/core/ncd/extract.py         22%     Estrazione embeddings NCD
app/core/supcon.py              13%     SupCon loss + training
app/preprocessing/cropper.py     0%     Faster R-CNN crop
app/preprocessing/split.py       0%     Train/test split
app/preprocessing/dataset_tool  28%     Utility dataset
app/explainability/*.py          0%     GradCAM heatmap
```

### Frontend: zero coverage

```
src/hooks/useSort.ts            0%     Sorting hook
src/components/ui/SortHeader.tsx 0%     Sortable header
src/features/explorer/*.tsx     0%     Matrix, Benchmark, KPI
src/pages/ExperimentsPage.tsx   0%     Experiments page
src/pages/*.tsx                 0%     Tutte le altre pagine
src/lib/api.ts                  0%     API client
src/stores/useAppStore.ts       0%     Zustand store
```

---

## 2. Strategia

### 2.1 Backend API — TestClient + fixture

Ogni route FastAPI viene testata con `TestClient`. Le fixture:
- Creano esperimenti finti su `tmp_path` per i test che leggono da file JSON
- Usano `monkeypatch` per reindirizzare `_get_experiments_dir()`
- Per auth: mockano il database SQLite con un DB temporaneo

Template (le parti tra `{}` sono segnaposto da sostituire):

```python
# tests/test_api/test_{route}.py
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

class Test{Endpoint}:
    """Tests for {method} {path}."""

    def test_{scenario}_returns_200(self, {fixtures}):
        """Happy path."""
        response = client.{method}("{path}")
        assert response.status_code == 200
        # Verifica campi obbligatori
        data = response.json()
        assert "expected_field" in data

    def test_{scenario}_invalid_input_returns_422(self):
        """Validation error."""
        response = client.{method}("{path}", json={"bad": "data"})
        assert response.status_code == 422

    def test_{scenario}_not_found_returns_404(self):
        """Missing resource."""
        response = client.{method}("{path}/nonexistent")
        assert response.status_code == 404

    def test_{scenario}_unauthorized_returns_401(self):
        """Auth required."""
        response = client.{method}("{path}")  # no auth header
        assert response.status_code == 401
```

### 2.2 Backend core — pytest + mock oggetti reali

Template per moduli core:

```python
# tests/test_core/test_{module}.py
import pytest
import torch
from app.core.{module} import {function}

class Test{Function}:
    def test_happy_path(self, {fixtures}):
        result = {function}(valid_input)
        assert result is not None
        assert isinstance(result, expected_type)

    def test_edge_case_empty(self):
        """Lista vuota / tensore zero / batch None."""
        result = {function}(empty_input)
        assert result == expected_fallback

    def test_edge_case_none(self):
        """Input null."""
        result = {function}(None)
        assert result is None

    def test_error_case_invalid(self):
        """Input malformato."""
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            {function}(invalid_input)
```

### 2.3 Frontend — vitest + testing-library

#### Funzioni pure (Fase 3)

Già iniziato con `sort.test.ts` e `metrics-colors.test.ts`. Estendere a tutte le funzioni in `types.ts`:

```typescript
// src/features/explorer/__tests__/types.test.ts
import { getArchitectureFamily, getExperimentSeries, getArchitectureLabel } from "../types";

describe("getArchitectureFamily", () => {
  it("returns ResNet for resnet18", () => {
    expect(getArchitectureFamily("resnet18")).toBe("ResNet");
  });
  it("returns null for unknown variant", () => {
    expect(getArchitectureFamily("unknown")).toBeNull();
  });
  it("returns null for null input", () => {
    expect(getArchitectureFamily(null)).toBeNull();
  });
});
```

#### Hooks (Fase 4)

`renderHook` + attore per testare transizioni di stato:

```typescript
// src/hooks/__tests__/useSort.test.ts
import { renderHook, act } from "@testing-library/react";
import { useSort } from "../useSort";

const items = [
  { id: "exp6", mean_f1: 0.85 },
  { id: "exp7", mean_f1: 0.92 },
  { id: "exp10", mean_f1: 0.78 },
];

describe("useSort", () => {
  it("sorts by default key ascending", () => {
    const { result } = renderHook(() => useSort(items, "id"));
    expect(result.current.sorted[0].id).toBe("exp10"); // '1' < '6' in string compare
  });

  it("toggles direction on same key", () => {
    const { result } = renderHook(() => useSort(items, "id"));
    act(() => result.current.toggle("id"));
    expect(result.current.dir).toBe("desc");
    expect(result.current.sorted[0].id).toBe("exp7"); // '7' > ...
  });

  it("resets to asc on different key", () => {
    const { result } = renderHook(() => useSort(items, "id"));
    act(() => result.current.toggle("mean_f1"));
    expect(result.current.dir).toBe("asc");
    expect(result.current.key).toBe("mean_f1");
  });
});
```

#### Componenti (Fase 5-6)

Mock di i18n + router, test con `render` + `screen`:

```typescript
// src/components/ui/__tests__/SortHeader.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SortHeader } from "../SortHeader";

const mockToggle = vi.fn();

describe("SortHeader", () => {
  it("renders label", () => {
    render(<SortHeader label="F1" sortKey="mean_f1" activeKey="id" dir="asc" onToggle={mockToggle} />);
    expect(screen.getByText("F1")).toBeInTheDocument();
  });

  it("shows sort indicator when active", () => {
    render(<SortHeader label="F1" sortKey="mean_f1" activeKey="mean_f1" dir="desc" onToggle={mockToggle} />);
    expect(screen.getByText("↓")).toBeInTheDocument();
  });

  it("calls onToggle on click", async () => {
    const user = userEvent.setup();
    render(<SortHeader label="F1" sortKey="mean_f1" activeKey="id" dir="asc" onToggle={mockToggle} />);
    await user.click(screen.getByText("F1"));
    expect(mockToggle).toHaveBeenCalledWith("mean_f1");
  });

  it("has correct aria-sort when active", () => {
    const { container } = render(
      <SortHeader label="F1" sortKey="mean_f1" activeKey="mean_f1" dir="asc" onToggle={mockToggle} />,
    );
    expect(container.querySelector("th")).toHaveAttribute("aria-sort", "ascending");
  });
});
```

---

## 3. Fasi di implementazione

### Fase 1 — Backend API endpoints mancanti

**PR #1**: Auth + Admin + Dataset + Detail + Embeddings

| File | Test |
|------|------|
| `tests/test_api/test_auth.py` | register (successo, username già preso, password debole), login (successo, credenziali errate), me (autenticato, non autenticato) |
| `tests/test_api/test_admin.py` | delete user (admin, superadmin, non autorizzato), list users |
| `tests/test_api/test_dataset.py` | summary (con dati, senza dati, specie novel inclusa), gallery |
| `tests/test_api/test_experiments_detail.py` | detail (esistente, non esistente, con fold, senza fold) |
| `tests/test_api/test_embeddings.py` | list (con embedding, senza, filtro per experiment) |

### Fase 2 — Backend core modules

**PR #2**: Auth + K-fold + SupCon + Preprocessing split/crop

| File | Test |
|------|------|
| `tests/test_core/test_auth.py` | password hashing, JWT encode/decode, role hierarchy, superadmin create |
| `tests/test_core/test_kfold_training.py` | training loop con mock data, fold completion, checkpoint save/load |
| `tests/test_core/test_supcon.py` | SupConLoss forward/backward, contrastive training step |
| `tests/test_preprocessing/test_split.py` | stratificato, MD5 dedup, hold-out extraction |
| `tests/test_preprocessing/test_cropper.py` | crop con Faster R-CNN mock, no detection fallback |

### Fase 3 — Frontend pure functions

**PR #3**: types.ts helpers + ExperimentsPage helpers (colLabel/colAlign)

| File | Test count |
|------|-----------|
| `src/features/explorer/__tests__/types.test.ts` | ~15 test |
| `src/pages/__tests__/ExperimentsPage.test.tsx` (helpers only) | ~5 test |

Nota: `tsconfig.app.json` exclude per `__tests__/` già fatto nella PR #85.

### Fase 4 — Frontend hooks

**PR #4**: useSort + useExperiments

| File | Test |
|------|------|
| `src/hooks/__tests__/useSort.test.ts` | sort asc/desc, toggle, reset, empty items, single item, null values |

### Fase 5 — Frontend components

**PR #5**: SortHeader + MatrixSection + BenchmarkMatrix

| File | Test |
|------|------|
| `src/components/ui/__tests__/SortHeader.test.tsx` | render, click, aria-sort, active/inactive |
| `src/features/explorer/__tests__/MatrixSection.test.tsx` | render con dati, render vuoto, hover tooltip, keyboard nav |
| `src/features/explorer/__tests__/BenchmarkMatrix.test.tsx` | render con lookup, filtri famiglia/serie |

### Fase 6 — Frontend pagine principali

**PR #6**: ExperimentsPage + DashboardPage + ClassifyPage

Le pagine sono i componenti più grandi (100-700 righe ciascuna). Testare:
- **ExperimentsPage** (~230 righe): loading, error, empty, matrix view, table view, sort click, filter select, compare selection
- **DashboardPage** (~290 righe): render KPIs, nav cards, NCD cards condizionali, stato vuoto
- **ClassifyPage** (~440 righe): upload, classify result, errore modello, loading prediction

Per ogni pagina:
1. Mock dipendenze (API, i18n, router)
2. Test stati: loading → empty → error → success
3. Test interazioni utente principali
4. Test rendering condizionale (sezioni visibili/nascoste)

| File | Test |
|------|------|
| `src/pages/__tests__/ExperimentsPage.test.tsx` | ~20 test |
| `src/pages/__tests__/DashboardPage.test.tsx` | ~15 test |
| `src/pages/__tests__/ClassifyPage.test.tsx` | ~15 test |

### Fase 7 — Frontend pagine rimanenti (se coverage < 70%)

**PR #7**: ResearchPage + EmbeddingsPage + DatasetPage + AuthPage + UserManagementPage

Test più leggeri per pagine meno critiche:
- **ResearchPage**: render sezioni, citation popover open/close
- **EmbeddingsPage**: render scatter plot, selezione esperimento
- **DatasetPage**: render tabs, split selection
- **AuthPage**: login form validation, register form, error display
- **UserManagementPage**: user list, delete confirmation, role display

| File | Test |
|------|------|
| `src/pages/__tests__/ResearchPage.test.tsx` | ~10 test |
| `src/pages/__tests__/EmbeddingsPage.test.tsx` | ~8 test |
| `src/pages/__tests__/DatasetPage.test.tsx` | ~8 test |
| `src/pages/__tests__/AuthPage.test.tsx` | ~10 test |
| `src/pages/__tests__/UserManagementPage.test.tsx` | ~8 test |

---

## 4. Mock strategy

### Backend

| Dipendenza | Mock |
|-----------|------|
| File system | `tmp_path` fixture + `monkeypatch` su `_get_experiments_dir()` |
| Database SQLite | `tempfile.mkstemp()` per DB temporaneo |
| PyTorch model | `unittest.mock.patch("torch.load")` |
| Token JWT | `AuthBypass` middleware nei test o `force_authenticate` fixture |
| Classify model | `unittest.mock.patch("app.core.classify.ModelManager")` |

### Frontend

| Dipendenza | Mock |
|-----------|------|
| i18n (`react-i18next`) | `vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (k: string) => k }) }))` |
| Router | `MemoryRouter` wrapper |
| API (`fetchExperiments`) | `vi.spyOn(api, "fetchExperiments").mockResolvedValue(...)` |
| `useNavigate` | `vi.fn()` |

---

## 5. Quality gates

Ogni PR deve superare:

```bash
# Backend
cd backend
pytest -x -q --cov=app --cov=scripts --cov-fail-under=70

# Frontend  
cd frontend
npm run build        # tsc + vite build
npm run test         # vitest
npm run check-i18n   # 100% coverage EN/IT
```

---

## 6. Stima effort

| Fase | Area | Nuovi test | Coverage after |
|------|------|-----------|----------------|
| 0 | Merge PR #89 | — | — |
| 1 | Backend API | ~40 | BE 40% → 44% |
| 2 | Backend core | ~60 | BE 44% → 65% |
| 3 | Frontend pure functions | ~20 | FE 1% → 13% |
| 4 | Frontend hooks | ~10 | FE 13% → 16% |
| 5 | Frontend components | ~20 | FE 16% → 40% |
| 6 | Frontend pagine principali | ~50 | FE 40% → 60% |
| 7 | Frontend pagine rimanenti | ~40 | FE 60% → 72% |

**Totale**: ~240 test, ~7 PR.
