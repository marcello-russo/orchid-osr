# Dashboard KPI Fix — Design Doc

## Objective

Fix dashboard KPIs: correct totals, add variance-awareness to F1, add Open-World KPIs.

## 1. Species endpoint — includere tutte le 7 specie

**Backend** `backend/app/api/routes/species.py:64`

Cambiare il filtro `valid` da `set(settings.class_names)` (6 classi) a includere anche la novel class:

```python
valid = set(settings.class_names) | {settings.novel_class_name}
```

Così la risposta include tutte e 7 le specie, sia nel runtime scan che nel fallback `species_counts.json`.

`total_species` → 7. Tabella distribuzione mostra tutte e 7.

## 2. species_counts.json — aggiornare conteggi reali

**File** `backend/app/data/species_counts.json`

Conteggi attuali dal dataset `cropped_256x512/` (train + test):

| Specie | Train | Test | Totale |
|--------|-------|------|--------|
| O. exaltata | 490 | 50 | 540 |
| O. garganica | 563 | 50 | 613 |
| O. incubacea | 352 | 50 | 402 |
| O. majellensis | 152 | 50 | 202 |
| O. sphegodes | 425 | 50 | 475 |
| O. sphegodes_Napoli | 0 | 548 | 548 |
| O. sphegodes_Palena | 210 | 50 | 260 |
| **Totale** | **2192** | **848** | **3040** |

## 3. KPI Classificazione (come ora, ma migliorato)

### Best Test Accuracy
- Nessun std disponibile (singola valutazione su test set)
- Selezione: max di `test_accuracy` tra modelli exp6
- Display: `XX.XX%`

### Best F1 (con varianza)
- `mean_f1` è la media tra fold, `std_f1` è la deviazione standard
- **Selezione**: max di `(mean_f1 - std_f1)` — lower bound pessimistico
- **Display**: `XX.XX% ± YY.YY%` (mean ± std)
- Se `std_f1` non disponibile, assume 0

**Frontend** `frontend/src/pages/DashboardPage.tsx`:
```tsx
const bestF1Model = cleanSplitExps.reduce((best, e) => {
  const score = (e.mean_f1 ?? 0) - (e.std_f1 ?? 0);
  const bestScore = (best.mean_f1 ?? 0) - (best.std_f1 ?? 0);
  return score > bestScore ? e : best;
}, cleanSplitExps[0]);
const bestF1Val = bestF1Model?.mean_f1 ?? 0;
const bestF1Std = bestF1Model?.std_f1 ?? 0;
```

## 4. KPI Open-World (nuova riga)

Tutti e 12 i modelli NCD (exp8 + exp10) hanno `ncd_auroc` e `ncd_clustering_nmi` disponibili.

| KPI | Fonte | Selezione | Display |
|-----|-------|-----------|---------|
| **Best NCD AUROC** | `ncd_auroc` | max del valore | `XX.XX%` |
| **Best NCD NMI** | `ncd_clustering_nmi` | max del valore | `X.XXX` |

Filtro: esperimenti dove `ncd_auroc is not None`. Visibile SOLO se almeno un modello NCD esiste.

Layout: seconda riga di KPI sotto quella esistente, con titolo "Open-World" e sfondo leggermente diverso.

## 5. Schema riassuntivo

```
Prima riga (Classificazione):
┌─────────────┬──────────────┬───────────────┬───────────────┬──────────┬─────────────┐
│ Experiments │ Modelli comp.│ Best Test Acc │ Best F1       │ Classi   │ Immagini    │
│ (conteggio) │ (conteggio)  │ XX.XX%        │ XX.XX%±YY.YY% │ 7        │ 3040        │
└─────────────┴──────────────┴───────────────┴───────────────┴──────────┴─────────────┘

Seconda riga (Open-World) — se presente almeno un NCD:
┌────────────────┬────────────────┐
│ Best NCD AUROC │ Best NCD NMI   │
│ XX.XX%         │ X.XXX          │
└────────────────┴────────────────┘
```

## Files da modificare

| File | Modifica |
|------|----------|
| `backend/app/data/species_counts.json` | Nuovi conteggi (3040 total, 7 species) |
| `backend/app/api/routes/species.py:64` | `valid` include `novel_class_name` |
| `frontend/src/pages/DashboardPage.tsx` | F1 con ±std e mean-std selection; NCD KPIs |

## Test

- Dashboard mostra 7 specie, totale 3040 immagini
- Best F1 mostra ± std e seleziona il modello corretto (mean-std più alto)
- NCD AUROC e NMI appaiono solo se ci sono modelli NCD
- Build TypeScript: `cd frontend && npx tsc --noEmit`
