# Experiments Page Redesign — Design Doc

## Objective

Replace the flat experiments table with a dual-view page: a **benchmark heatmap matrix** (primary) and an **interactive detail table** (secondary). Add architecture-based filtering, inline comparison, and OS-friendly visual hierarchy.

## Data Model

The API already returns `ExperimentSummary` with all needed fields:
- `id`, `model_variant` (resnet18, dinov2_base, etc.), `mean_f1`, `std_f1`, `test_accuracy`
- `ncd_auroc`, `ncd_clustering_nmi` (null if not applicable)
- `is_ncd`, `type`, `folds_completed`/`num_folds`

Family mapping (hardcoded in frontend):
```
resnet18 / resnet50 → "ResNet"
convnext_tiny / convnext_small → "ConvNeXt"
dinov2_small / dinov2_base → "DINOv2"
```

Experiment series derived from `id` prefix:
```
exp6_* → "Exp 6 (CE)"
exp7_* → "Exp 7 (SupCon)"
exp8_* → "Exp 8 (SpHOR)"
exp10_* → "Exp 10 (MGH-NCD)"
```

## Views

### View 1: Benchmark Matrix (default)

A 6×4 heatmap: rows = architectures, columns = experiment series.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  [All] [ResNet] [ConvNeXt] [DINOv2]   [Exp 6] [Exp 7] [Exp 8] [Exp 10]  │
│                                                                     │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐               │
│  │         │ Exp 6 CE│Exp 7 SuC│Exp 8 SpH│Exp10 MGH│               │
│  ├─────────┼─────────┼─────────┼─────────┼─────────┤               │
│  │ResNet-18│  0.880  │  0.736  │  0.520  │  0.226  │               │
│  │ResNet-50│  0.882  │  0.799  │  0.609  │  0.174  │               │
│  │Conv-Tiny│  0.890  │  0.681  │  ★0.759 │  0.349  │               │
│  │Conv-Sml │  0.863  │  0.689  │  0.668  │  0.247  │               │
│  │DINOv2-S │  0.888  │★ 0.896  │  0.747  │  0.270  │               │
│  │DINOv2-B │★ 0.927  │  0.687  │  0.654  │  0.369  │               │
│  └─────────┴─────────┴─────────┴─────────┴─────────┘               │
│                                                                     │
│  OSR Detection Rate:                                                │
│  ┌─────────┬─────────┬─────────┐                                    │
│  │         │ Exp 8   │ Exp 10  │                                    │
│  ├─────────┼─────────┼─────────┤                                    │
│  │Conv-Tiny│ ★0.907  │  0.566  │                                    │
│  │DINOv2-B │  0.746  │  0.688  │                                    │
│  └─────────┴─────────┴─────────┘                                    │
│                                                                     │
│  [View detail table →]                                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Behavior:**
- Each cell shows F1 test value with green-to-red background
- ★ marks the best value in each column
- Hover on cell → tooltip with `F1 CV`, `std`, `test_accuracy`
- Click on cell → navigate to `/experiments/{id}`
- Architecture filter buttons above matrix filter rows
- Experiment series filter buttons filter columns
- OSR table shows only NCD experiments with `ncd_auroc`

**Implementation:**
- Custom `<BenchmarkMatrix>` component
- Rows: 6 architecture labels with color indicators
- Colored cells using `oklch()` or Tailwind opacity based on F1 value
- No external chart library needed — pure HTML/CSS grid

### View 2: Detail Table

Toggle from matrix via "View detail table" button.

```
┌──────────────────────────────────────────────────────────────────────┐
│  [Matrix view]  [Detail table]                                       │
│                                                                      │
│  Filter: [All architectures ▼]  [All experiments ▼]                  │
│                                                                      │
│  ┌──────┬──────────┬────────┬──────┬──────────┬──────────┬────────┐ │
│  │  ☐  │ Experiment│ Model  │ F1   │ ±std     │ Test Acc │  NCD   │ │
│  ├──────┼──────────┼────────┼──────┼──────────┼──────────┼────────┤ │
│  │  ☐  │ Exp 6 CE │ ResNet…│0.880 │ ±0.023   │ 97.49%   │   —    │ │
│  │  ☐  │ Exp 8 SpH│ ConvNe…│0.759 │ ±0.031   │ 81.47%   │ 0.907  │ │
│  └──────┴──────────┴────────┴──────┴──────────┴──────────┴────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

**Columns:** checkbox, Experiment (with family badge), Model (mono), F1 (with ±std), Test Acc, NCD AUROC (or —), Folds completed

**Features:**
- Click column header → sort ascending/descending
- Architecture filter dropdown
- Experiment series filter dropdown
- Checkbox selection (up to 4) → "Compare" button shows inline comparison panel
- Family badge colored (blue=ResNet, orange=ConvNeXt, purple=DINOv2)

**Inline Comparison Panel:**
When 2+ experiments selected, a collapsible div appears below the table showing the `CompareModeView` component (already exists — reuse).

## Components

### New: `BenchmarkMatrix.tsx`
- Receives `experiments: ExperimentSummary[]`
- Renders the 6×4 heatmap grid
- Architecture/experiment filter buttons
- OSR sub-table
- Cell click handler → navigate

### Modified: `ExperimentsPage.tsx`
- Add state for active view (`"matrix"` | `"table"`)
- Conditionally render `BenchmarkMatrix` or existing table
- Add architecture filter + experiment series filter to table
- Add sortable columns to table
- Add family badges
- Add ±std display

## Files

| File | Change |
|------|--------|
| `frontend/src/pages/ExperimentsPage.tsx` | Major rewrite: dual view, filters, sorting |
| `frontend/src/features/explorer/components/BenchmarkMatrix.tsx` | **New** — heatmap grid |
| `frontend/src/features/explorer/types.ts` | Add helper functions for family/series |
| `frontend/src/lib/types.ts` | Maybe add `family` and `series` helpers |
| `frontend/src/locales/en.json` | New keys for matrix |
| `frontend/src/locales/it.json` | New keys for matrix |

## Test

- Matrix shows all 24 experiments in 6×4 grid
- Architecture filter hides/shows rows correctly
- Experiment filter hides/shows columns correctly
- OSR table shows only exp8 and exp10
- Tooltip on hover shows detail
- Click on cell navigates to detail page
- Detail table sorts by column click
- Detail table filters work
- Compare panel shows inline when 2+ selected
