# Dashboard KPI Fix — Design Doc

## Objective

Fix dashboard KPIs: correct total images/classes count, improve Best F1 with variance awareness, add Open-World KPIs.

## Changes

### 1. Species endpoint — include all 7 species

**Backend** `backend/app/api/routes/species.py`

- Remove the filter that excludes species not in `settings.class_names`
- Instead, include ALL species found in dataset directories (or in `species_counts.json`), including `O. sphegodes_Napoli`
- `total_species` changes from 6 → 7
- Species distribution table shows all 7 species

### 2. Update species_counts.json

`backend/app/data/species_counts.json`

- Update counts to reflect current dataset state (after 40 duplicate removal)
- Keep all 7 species

### 3. Best Test Accuracy

No std available (single test set evaluation). Keep current behavior: max of `test_accuracy` across exp6 models. Display as `XX.XX%`.

### 4. Best F1 — lower bound selection + std display

**Frontend** `frontend/src/pages/DashboardPage.tsx`

- Selection criterion: `mean_f1 - std_f1` (lower bound pessimistic)
- The model with the highest `mean_f1 - std_f1` wins
- Display: `XX.XX% ± YY.YY%`
- If `std_f1` is not available for a model, use 0

```tsx
const bestF1Model = cleanSplitExps.reduce((best, e) => {
  const score = (e.mean_f1 ?? 0) - (e.std_f1 ?? 0);
  const bestScore = (best.mean_f1 ?? 0) - (best.std_f1 ?? 0);
  return score > bestScore ? e : best;
}, cleanSplitExps[0]);
```

### 5. Open-World KPIs (second row)

Add a second KPI row below the classification row, visible only if NCD experiments exist with data.

| KPI | Source | Description |
|-----|--------|-------------|
| **Best NCD AUROC** | `ncd_auroc` from experiments | Max AUROC across exp8/10 models (open-set detection) |
| **Best NCD NMI** | `ncd_clustering_nmi` from experiments | Max NMI across exp8/10 models (novel class discovery) |

Selection: max of the metric value.

Layout: new row with 2 KPI cards, title "Open-World Detection" or similar label.

### Frontend data flow

```
exp6 models → filter exp6 kfold → bestF1 = max(mean - std) → "XX.XX% ± YY.YY%"
ncd models → filter exp8/10 → bestAUROC = max(ncd_auroc) → "XX.XX%"
ncd models → filter exp8/10 → bestNMI = max(ncd_clustering_nmi) → "X.XXX"
species → include all 7 → "7 species", "XXXX total images"
```

### Files to modify

| File | Change |
|------|--------|
| `backend/app/data/species_counts.json` | Update counts with current dataset |
| `backend/app/api/routes/species.py` | Remove `valid` filter, include all species |
| `frontend/src/pages/DashboardPage.tsx` | Fix F1 selection (mean-std), add ±std display, add NCD KPI row |

### Testing

- Verify dashboard shows 7 species
- Verify total images count is correct
- Verify Best F1 selection picks the right model (highest mean-std)
- Verify Best F1 shows ± std
- Verify NCD AUROC and NMI appear in new row
- Verify existing models (ResNet, ConvNeXt, DINOv2) all still work
