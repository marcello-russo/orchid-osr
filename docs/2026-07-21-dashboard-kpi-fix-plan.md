# Dashboard KPI Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix dashboard KPIs: correct totals (3040 images, 7 species), add variance to F1, add Open-World KPIs.

**Architecture:** Backend changes (species counts + filter) then frontend changes (F1 logic, NCD KPIs, i18n).

**Tech Stack:** Python/FastAPI + React/TypeScript/Tailwind + Zustand + i18next

---

### Task 1: Species backend — aggiornare conteggi e filtro

**Files:**
- Modify: `backend/app/data/species_counts.json`
- Modify: `backend/app/api/routes/species.py:64`

- [ ] **Step 1: Aggiornare species_counts.json**

Sostituire il contenuto con i conteggi reali (verificati sul dataset `cropped_256x512/`):

```json
{
  "species": {
    "O. exaltata": 540,
    "O. garganica": 613,
    "O. incubacea": 402,
    "O. majellensis": 202,
    "O. sphegodes": 475,
    "O. sphegodes_Napoli": 548,
    "O. sphegodes_Palena": 260
  },
  "total_images": 3040,
  "total_species": 7
}
```

- [ ] **Step 2: Cambiare il filtro valid in species.py**

Modificare `backend/app/api/routes/species.py` riga 64:

```python
# Prima:
valid = set(settings.class_names)
# Dopo:
valid = set(settings.class_names) | {settings.novel_class_name}
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/data/species_counts.json backend/app/api/routes/species.py
git commit -m "fix: species endpoint include tutte e 7 le specie, conteggi aggiornati"
```

---

### Task 2: i18n — aggiungere chiavi per NCD KPIs

**Files:**
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/it.json`

- [ ] **Step 1: Aggiungere chiavi in en.json**

All'interno di `"kpis": { ... }` aggiungere:

```json
"best_ncd_auroc": "Best NCD AUROC",
"best_ncd_nmi": "Best NCD NMI"
```

Le chiavi kpis esistenti sono: `total_images`, `classes`, `completed_models`, `best_f1`, `best_test_accuracy`, `in_dataset`, `combined_dataset`.

Inserire le nuove righe dopo `combined_dataset`.

- [ ] **Step 2: Aggiungere chiavi in it.json**

```json
"best_ncd_auroc": "Miglior AUROC NCD",
"best_ncd_nmi": "Miglior NMI NCD"
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/locales/en.json frontend/src/locales/it.json
git commit -m "feat: i18n keys per NCD KPIs"
```

---

### Task 3: TypeScript — aggiungere campi NCD a ExperimentSummary

**Files:**
- Modify: `frontend/src/lib/types.ts:86-98`

- [ ] **Step 1: Aggiungere i campi mancanti**

Aggiungere dopo `std_f1` e prima della chiusura `}`:

```typescript
  is_ncd: boolean;
  ncd_auroc: number | null;
  ncd_clustering_nmi: number | null;
```

Il risultato finale:

```typescript
export interface ExperimentSummary {
  id: string;
  name: string;
  model_variant: string | null;
  type: "baseline" | "kfold";
  status: "completed" | "partial";
  num_folds: number | null;
  folds_completed: number | null;
  test_accuracy: number | null;
  mean_f1: number | null;
  std_f1: number | null;
  timestamp: string | null;
  is_ncd: boolean;
  ncd_auroc: number | null;
  ncd_clustering_nmi: number | null;
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/lib/types.ts
git commit -m "feat: aggiunti campi NCD a ExperimentSummary type"
```

---

### Task 4: Dashboard — Best F1 con ±std e lower bound selection

**Files:**
- Modify: `frontend/src/pages/DashboardPage.tsx`

- [ ] **Step 1: Modificare la selezione e visualizzazione Best F1**

Sostituire il calcolo di `bestMeanF1` (righe 41-44) con:

```tsx
const bestF1Model = cleanSplitExps.length > 0
  ? cleanSplitExps.reduce((best, e) => {
      const score = (e.mean_f1 ?? 0) - (e.std_f1 ?? 0);
      const bestScore = (best.mean_f1 ?? 0) - (best.std_f1 ?? 0);
      return score > bestScore ? e : best;
    })
  : null;
const bestF1Val = bestF1Model?.mean_f1 ?? 0;
const bestF1Std = bestF1Model?.std_f1 ?? 0;
```

Sostituire il KpiCard del Best F1 (riga 130-134) per mostrare ±std:

```tsx
<KpiCard
  title={t("dashboard.kpis.best_f1")}
  value={bestF1Val > 0 ? `${(bestF1Val * 100).toFixed(2)}% ± ${(bestF1Std * 100).toFixed(2)}%` : "—"}
  icon={BarChart3}
  delay={150}
/>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/DashboardPage.tsx
git commit -m "fix: Best F1 usa mean-std per selezione, mostra ±std"
```

---

### Task 5: Dashboard — NCD KPIs (seconda riga)

**Files:**
- Modify: `frontend/src/pages/DashboardPage.tsx`

- [ ] **Step 1: Calcolare metriche NCD**

Dopo `bestF1Std`, aggiungere il calcolo delle metriche NCD:

```tsx
const ncdExps = (experiments ?? []).filter((e) => e.ncd_auroc != null);
const bestNcdAuroc = ncdExps.length > 0
  ? ncdExps.reduce((max, e) => Math.max(max, e.ncd_auroc!), 0)
  : 0;
const bestNcdNmi = ncdExps.length > 0
  ? ncdExps.reduce((max, e) => Math.max(max, e.ncd_clustering_nmi ?? 0), 0)
  : 0;
```

- [ ] **Step 2: Aggiungere la seconda riga KPI**

Dopo il `<div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">` esistente (riga 111), aggiungere:

```tsx
{ncdExps.length > 0 && (
  <div>
    <h2 className="text-lg font-semibold mb-4">{t("dashboard.kpis.open_world")}</h2>
    <div className="grid grid-cols-2 gap-3">
      <KpiCard
        title={t("dashboard.kpis.best_ncd_auroc")}
        value={bestNcdAuroc > 0 ? `${(bestNcdAuroc * 100).toFixed(2)}%` : "—"}
        icon={ScanSearch}
        delay={300}
      />
      <KpiCard
        title={t("dashboard.kpis.best_ncd_nmi")}
        value={bestNcdNmi > 0 ? bestNcdNmi.toFixed(4) : "—"}
        icon={BarChart3}
        delay={350}
      />
    </div>
  </div>
)}
```

Nota: aggiungere `ScanSearch` all'import da `lucide-react` se non già presente (dovrebbe già esserci alla riga 14).

Nota: aggiungere la chiave i18n `open_world` in en.json e it.json:
- `"open_world": "Open-World Detection"`
- `"open_world": "Rilevamento Open-World"`

Aggiungere le chiavi nel blocco `"kpis"`.

- [ ] **Step 3: Aggiungere chiave open_world in i18n**

In `frontend/src/locales/en.json` e `frontend/src/locales/it.json`, nel blocco `"kpis"`:

```json
"open_world": "Open-World Detection"
```

```json
"open_world": "Rilevamento Open-World"
```

- [ ] **Step 4: Build e verifica**

```bash
cd frontend && npx tsc --noEmit
```

Nessun errore TypeScript.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/DashboardPage.tsx frontend/src/locales/en.json frontend/src/locales/it.json
git commit -m "feat: dashboard NCD KPIs (AUROC + NMI)"
```

- [ ] **Step 6: Push**

```bash
git push origin main
```

---

### Riassunto modifiche per file

| File | Task | Tipo modifica |
|------|------|---------------|
| `backend/app/data/species_counts.json` | 1 | Sovrascrittura |
| `backend/app/api/routes/species.py` | 1 | 1 riga |
| `frontend/src/locales/en.json` | 2, 5 | 4 righe aggiunte |
| `frontend/src/locales/it.json` | 2, 5 | 4 righe aggiunte |
| `frontend/src/lib/types.ts` | 3 | 3 righe aggiunte a ExperimentSummary |
| `frontend/src/pages/DashboardPage.tsx` | 4, 5 | ~30 righe modificate/aggiunte |
