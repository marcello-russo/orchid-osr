# Experiments Page Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat experiments table with dual-view page: benchmark heatmap matrix (default) + interactive detail table with sorting, filtering, and inline comparison.

**Architecture:** New `BenchmarkMatrix` component renders 6×4 F1 heatmap + 6×2 OSR sub-table. `ExperimentsPage.tsx` rewritten with dual-view toggle, architecture/experiment filters, sortable columns, family badges. Helper functions in `types.ts` for family/series mapping.

**Tech Stack:** React + TypeScript + Tailwind + shadcn/ui Table

---

### Task 1: Helper functions — family, series, color

**Files:**
- Modify: `frontend/src/features/explorer/types.ts`

- [ ] **Step 1: Add helper functions**

At the end of `frontend/src/features/explorer/types.ts` (after `foldToEpochPoints`), add:

```typescript
export type ArchitectureFamily = "ResNet" | "ConvNeXt" | "DINOv2";
export type ExperimentSeries = "Exp 6" | "Exp 7" | "Exp 8" | "Exp 10";

const FAMILY_MAP: Record<string, ArchitectureFamily> = {
  resnet18: "ResNet", resnet50: "ResNet",
  convnext_tiny: "ConvNeXt", convnext_small: "ConvNeXt",
  dinov2_small: "DINOv2", dinov2_base: "DINOv2",
};

const SERIES_MAP: Record<string, ExperimentSeries> = {
  exp6: "Exp 6", exp7: "Exp 7", exp8: "Exp 8", exp10: "Exp 10",
};

export function getArchitectureFamily(variant: string | null): ArchitectureFamily | null {
  return variant ? (FAMILY_MAP[variant] ?? null) : null;
}

export function getExperimentSeries(id: string): ExperimentSeries | null {
  const prefix = id.split("_")[0];
  return SERIES_MAP[prefix] ?? null;
}

export const FAMILY_COLORS: Record<ArchitectureFamily, string> = {
  ResNet: "#3B82F6",
  ConvNeXt: "#D97706",
  DINOv2: "#8B5CF6",
};

export const ARCHITECTURE_ORDER = [
  "ResNet-18", "ResNet-50",
  "ConvNeXt-Tiny", "ConvNeXt-Small",
  "DINOv2-Small", "DINOv2-Base",
];

export function getArchitectureLabel(variant: string | null): string {
  const map: Record<string, string> = {
    resnet18: "ResNet-18", resnet50: "ResNet-50",
    convnext_tiny: "ConvNeXt-Tiny", convnext_small: "ConvNeXt-Small",
    dinov2_small: "DINOv2-Small", dinov2_base: "DINOv2-Base",
  };
  return variant ? (map[variant] ?? variant) : "—";
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/features/explorer/types.ts
git commit -m "feat: helper functions for architecture family/series/label"
```

---

### Task 2: BenchmarkMatrix component

**Files:**
- Create: `frontend/src/features/explorer/components/BenchmarkMatrix.tsx`

- [ ] **Step 1: Create BenchmarkMatrix.tsx**

```tsx
import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import type { ExperimentSummary } from "@/lib/types";
import {
  getArchitectureFamily, getExperimentSeries, getArchitectureLabel,
  ARCHITECTURE_ORDER, FAMILY_COLORS,
} from "@/features/explorer/types";

interface BenchmarkMatrixProps {
  experiments: ExperimentSummary[];
  familyFilter: string | null;        // "ResNet" | "ConvNeXt" | "DINOv2" | null (all)
  seriesFilter: string | null;        // "Exp 6" | "Exp 7" | "Exp 8" | "Exp 10" | null (all)
}

const MATRIX_SERIES = ["Exp 6", "Exp 7", "Exp 8", "Exp 10"];
const OSR_SERIES = ["Exp 8", "Exp 10"];

function f1Color(f1: number): string {
  // Lightness: 0.15 (red) → 0.75 (green), clamped to [0, 1]
  const l = 0.15 + f1 * 0.6;
  return `oklch(${l} 0.15 ${f1 > 0.6 ? 145 : 25})`;
}

export function BenchmarkMatrix({ experiments, familyFilter, seriesFilter }: BenchmarkMatrixProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();

  // Build lookup: model_variant → { experiment_series → ExperimentSummary }
  const lookup = useMemo(() => {
    const map: Record<string, Record<string, ExperimentSummary>> = {};
    for (const exp of experiments) {
      const variant = exp.model_variant;
      const series = getExperimentSeries(exp.id);
      if (!variant || !series) continue;
      if (!map[variant]) map[variant] = {};
      map[variant][series] = exp;
    }
    return map;
  }, [experiments]);

  // Determine which columns to show
  const visibleSeries = seriesFilter ? [seriesFilter] : MATRIX_SERIES;
  const visibleOSR = seriesFilter
    ? (OSR_SERIES.includes(seriesFilter) ? [seriesFilter] : [])
    : OSR_SERIES;

  // Mapping from display label → model_variant string
  const labelToVariant: Record<string, string> = {
    "ResNet-18": "resnet18", "ResNet-50": "resnet50",
    "ConvNeXt-Tiny": "convnext_tiny", "ConvNeXt-Small": "convnext_small",
    "DINOv2-Small": "dinov2_small", "DINOv2-Base": "dinov2_base",
  };

  return (
    <div className="space-y-8">
      {/* F1 Matrix */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left py-2 pr-4 font-medium text-muted-foreground text-xs uppercase tracking-wider">
                {t("experiments.table.model")}
              </th>
              {visibleSeries.map((s) => (
                <th key={s} className="text-center py-2 px-3 font-medium text-xs uppercase tracking-wider text-muted-foreground">
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border/20">
            {ARCHITECTURE_ORDER.map((archLabel) => {
              const variant = labelToVariant[archLabel];
              const family = variant ? getArchitectureFamily(variant) : null;
              if (familyFilter && family !== familyFilter) return null;
              return (
                <tr key={archLabel} className="hover:bg-muted/20 transition-colors">
                  <td className="py-2 pr-4 font-medium text-sm flex items-center gap-2">
                    <span className="inline-block size-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: family ? FAMILY_COLORS[family] : "gray" }} />
                    <span>{archLabel}</span>
                  </td>
                  {visibleSeries.map((series) => {
                    const exp = lookup[variant]?.[series];
                    const f1 = exp?.mean_f1 ?? null;
                    const isBest = f1 !== null && f1 === Math.max(
                      ...ARCHITECTURE_ORDER
                        .map((l) => lookup[labelToVariant[l]]?.[series]?.mean_f1 ?? 0)
                    );
                    return (
                      <td key={series}
                        onClick={() => exp && navigate(`/experiments/${exp.id}`)}
                        className={`text-center py-2 px-3 rounded-lg cursor-pointer transition-all hover:scale-105 ${f1 !== null ? "text-white font-bold" : "text-muted-foreground"}`}
                        style={{ backgroundColor: f1 !== null ? f1Color(f1) : undefined }}
                        title={exp ? `F1 CV: ${((exp.mean_f1 ?? 0) * 100).toFixed(2)}%\nStd: ±${((exp.std_f1 ?? 0) * 100).toFixed(2)}%\nTest Acc: ${((exp.test_accuracy ?? 0) * 100).toFixed(2)}%` : "—"}
                      >
                        {f1 !== null ? `${(f1 * 100).toFixed(1)}` : "—"}
                        {isBest && <span className="ml-0.5 text-yellow-300">★</span>}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* OSR Sub-table */}
      {visibleOSR.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            {t("experiments.matrix.osr_title")}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="text-left py-2 pr-4 font-medium text-muted-foreground text-xs uppercase tracking-wider"></th>
                  {visibleOSR.map((s) => (
                    <th key={s} className="text-center py-2 px-3 font-medium text-xs uppercase tracking-wider text-muted-foreground">
                      {s}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border/20">
                {ARCHITECTURE_ORDER.map((archLabel) => {
                  const variant = labelToVariant[archLabel];
                  const family = variant ? getArchitectureFamily(variant) : null;
                  if (familyFilter && family !== familyFilter) return null;
                  return (
                    <tr key={archLabel} className="hover:bg-muted/20 transition-colors">
                      <td className="py-2 pr-4 font-medium text-sm flex items-center gap-2">
                        <span className="inline-block size-2.5 rounded-full shrink-0"
                          style={{ backgroundColor: family ? FAMILY_COLORS[family] : "gray" }} />
                        <span>{archLabel}</span>
                      </td>
                      {visibleOSR.map((series) => {
                        const exp = lookup[variant]?.[series];
                        const osr = exp?.ncd_auroc ?? null;
                        const isBest = osr !== null && osr === Math.max(
                          ...ARCHITECTURE_ORDER
                            .map((l) => lookup[labelToVariant[l]]?.[series]?.ncd_auroc ?? 0)
                        );
                        return (
                          <td key={series}
                            onClick={() => exp && navigate(`/experiments/${exp.id}`)}
                            className={`text-center py-2 px-3 rounded-lg cursor-pointer transition-all hover:scale-105 ${osr !== null ? "text-white font-bold" : "text-muted-foreground"}`}
                            style={{ backgroundColor: osr !== null ? f1Color(osr) : undefined }}
                          >
                            {osr !== null ? `${(osr * 100).toFixed(1)}` : "—"}
                            {isBest && <span className="ml-0.5 text-yellow-300">★</span>}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/features/explorer/components/BenchmarkMatrix.tsx
git commit -m "feat: BenchmarkMatrix component with F1 heatmap and OSR sub-table"
```

---

### Task 3: ExperimentsPage — dual view with matrix, table, filters, sorting

**Files:**
- Modify: `frontend/src/pages/ExperimentsPage.tsx`

- [ ] **Step 1: Riscrivere ExperimentsPage.tsx**

Sostituire TUTTO il contenuto del file con:

```tsx
import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { FlaskConical, Table2, Grid3X3 } from "lucide-react";
import { useExperiments } from "@/features/explorer/hooks/useExperiments";
import { BenchmarkMatrix } from "@/features/explorer/components/BenchmarkMatrix";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  getArchitectureFamily, getExperimentSeries, getArchitectureLabel,
  FAMILY_COLORS,
} from "@/features/explorer/types";

const ALL_FAMILIES = ["ResNet", "ConvNeXt", "DINOv2"] as const;
const ALL_SERIES = ["Exp 6", "Exp 7", "Exp 8", "Exp 10"] as const;
type SortKey = "id" | "model_variant" | "mean_f1" | "test_accuracy" | "ncd_auroc";
type SortDir = "asc" | "desc";

export function ExperimentsPage() {
  const { t } = useTranslation();
  const { data: experiments, isLoading } = useExperiments();

  // View state
  const [view, setView] = useState<"matrix" | "table">("matrix");
  const [familyFilter, setFamilyFilter] = useState<string | null>(null);
  const [seriesFilter, setSeriesFilter] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [selected, setSelected] = useState<Set<string>>(new Set());

  // Filter by exp9 excluded (already handled by backend, but safety check)
  const filtered = useMemo(() => {
    if (!experiments) return [];
    return experiments.filter((e) => {
      if (e.id.includes("exp9")) return false;
      if (familyFilter) {
        const fam = getArchitectureFamily(e.model_variant);
        if (fam !== familyFilter) return false;
      }
      if (seriesFilter) {
        const ser = getExperimentSeries(e.id);
        if (ser !== seriesFilter) return false;
      }
      return true;
    });
  }, [experiments, familyFilter, seriesFilter]);

  // Sort
  const sorted = useMemo(() => {
    const copy = [...filtered];
    copy.sort((a, b) => {
      const aVal = a[sortKey] ?? "";
      const bVal = b[sortKey] ?? "";
      const cmp = typeof aVal === "number" ? aVal - (bVal as number) : String(aVal).localeCompare(String(bVal));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [filtered, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("asc"); }
  };

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < 4) next.add(id);
      return next;
    });
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="size-10 md:size-12 rounded-2xl bg-purple-500/10 flex items-center justify-center shrink-0 text-purple-500">
          <FlaskConical className="size-6" />
        </div>
        <div>
          <h1 className="text-2xl md:text-4xl font-black tracking-tight">{t("experiments.title")}</h1>
          <p className="text-muted-foreground text-sm mt-1">{t("experiments.subtitle")}</p>
        </div>
      </div>

      {/* View toggle + filters */}
      <div className="flex flex-wrap gap-3 items-center justify-between">
        <div className="flex gap-2">
          <Button variant={view === "matrix" ? "default" : "outline"} size="sm" onClick={() => setView("matrix")}>
            <Grid3X3 className="size-4 mr-1" /> Matrix
          </Button>
          <Button variant={view === "table" ? "default" : "outline"} size="sm" onClick={() => setView("table")}>
            <Table2 className="size-4 mr-1" /> Table
          </Button>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          {/* Architecture filter */}
          <select value={familyFilter ?? ""} onChange={(e) => setFamilyFilter(e.target.value || null)}
            className="text-xs bg-card border border-border/50 rounded-lg px-2.5 py-1.5 text-foreground cursor-pointer appearance-auto">
            <option value="">{t("experiments.filters.all_architectures")}</option>
            {ALL_FAMILIES.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          {/* Experiment series filter */}
          <select value={seriesFilter ?? ""} onChange={(e) => setSeriesFilter(e.target.value || null)}
            className="text-xs bg-card border border-border/50 rounded-lg px-2.5 py-1.5 text-foreground cursor-pointer appearance-auto">
            <option value="">{t("experiments.filters.all_experiments")}</option>
            {ALL_SERIES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      ) : !experiments || experiments.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">
          <FlaskConical className="size-12 mx-auto mb-4 opacity-30" />
          <p>{t("experiments.empty")}</p>
        </div>
      ) : view === "matrix" ? (
        <>
          <BenchmarkMatrix
            experiments={experiments}
            familyFilter={familyFilter}
            seriesFilter={seriesFilter}
          />
          {/* Compare button for selected experiments */}
          {selected.size >= 2 && (
            <div className="flex justify-center">
              <Link
                to={`/compare?ids=${Array.from(selected).join(",")}`}
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
              >
                {t("experiments.compare")} ({selected.size})
              </Link>
            </div>
          )}
        </>
      ) : (
        <>
          {/* Detail Table */}
          <div className="rounded-lg border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10"><input type="checkbox" disabled /></TableHead>
                  <TableHead className="cursor-pointer select-none" onClick={() => toggleSort("id")}>
                    {t("experiments.table.experiment")} {sortKey === "id" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </TableHead>
                  <TableHead className="cursor-pointer select-none" onClick={() => toggleSort("model_variant")}>
                    {t("experiments.table.model")} {sortKey === "model_variant" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </TableHead>
                  <TableHead className="text-right cursor-pointer select-none" onClick={() => toggleSort("mean_f1")}>
                    F1 {sortKey === "mean_f1" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </TableHead>
                  <TableHead className="text-right cursor-pointer select-none" onClick={() => toggleSort("test_accuracy")}>
                    {t("experiments.table.test_acc")} {sortKey === "test_accuracy" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </TableHead>
                  <TableHead className="text-right cursor-pointer select-none" onClick={() => toggleSort("ncd_auroc")}>
                    NCD AUROC {sortKey === "ncd_auroc" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </TableHead>
                  <TableHead className="text-right">{t("experiments.table.folds")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.map((exp) => {
                  const family = getArchitectureFamily(exp.model_variant);
                  const series = getExperimentSeries(exp.id);
                  return (
                    <TableRow key={exp.id} className={selected.has(exp.id) ? "bg-primary/5" : ""}>
                      <TableCell>
                        <input type="checkbox" checked={selected.has(exp.id)}
                          onChange={() => toggleSelect(exp.id)} />
                      </TableCell>
                      <TableCell>
                        <Link to={`/experiments/${exp.id}`} className="text-primary hover:underline font-medium text-sm">
                          {series ? `${series}: ` : ""}{exp.name}
                        </Link>
                        {exp.is_ncd && <Badge variant="secondary" className="ml-2 text-[10px]">NCD</Badge>}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {family && (
                            <span className="inline-block size-2.5 rounded-full shrink-0"
                              style={{ backgroundColor: FAMILY_COLORS[family] }} />
                          )}
                          <span className="font-mono text-sm">{getArchitectureLabel(exp.model_variant)}</span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm tabular-nums">
                        {exp.mean_f1 != null ? `${(exp.mean_f1 * 100).toFixed(2)}%` : "—"}
                        {exp.std_f1 != null && (
                          <span className="text-muted-foreground text-xs ml-1">
                            ±{(exp.std_f1 * 100).toFixed(2)}%
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm tabular-nums">
                        {exp.test_accuracy != null ? `${(exp.test_accuracy * 100).toFixed(2)}%` : "—"}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm tabular-nums">
                        {exp.ncd_auroc != null ? `${(exp.ncd_auroc * 100).toFixed(2)}%` : "—"}
                      </TableCell>
                      <TableCell className="text-right text-sm text-muted-foreground">
                        {exp.folds_completed}/{exp.num_folds}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
          {/* Compare button */}
          {selected.size >= 2 && (
            <div className="flex justify-center">
              <Link
                to={`/compare?ids=${Array.from(selected).join(",")}`}
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
              >
                {t("experiments.compare")} ({selected.size})
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Build verification**

```bash
cd frontend && npx tsc --noEmit
```
Nessun errore.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/ExperimentsPage.tsx
git commit -m "feat: ExperimentsPage dual view (matrix/table), filters, sorting"
```

---

### Task 4: i18n keys

**Files:**
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/it.json`

Le chiavi `filters.all`, `filters.kfold`, `filters.baseline` già esistono. Aggiungere SOLO le nuove chiavi all'interno del blocco `"experiments"` esistente, unendo nel `filters` già presente e aggiungendo il nuovo blocco `matrix`:

- [ ] **Step 1: Aggiungere chiavi a en.json**

Nel blocco `"experiments"` esistente, aggiungere `"all_architectures"` e `"all_experiments"` dentro l'oggetto `filters` già presente, e un nuovo oggetto `matrix`:

```json
// dentro "experiments", merge in "filters":
"all_architectures": "All architectures",
"all_experiments": "All experiments"

// nuovo blocco sibling di "filters" e "table":
"matrix": {
  "osr_title": "OSR Detection Rate (@10% FPR)"
}
```

- [ ] **Step 2: Aggiungere chiavi a it.json**

```json
// dentro "experiments", merge in "filters":
"all_architectures": "Tutte le architetture",
"all_experiments": "Tutti gli esperimenti"

// nuovo blocco sibling di "filters" e "table":
"matrix": {
  "osr_title": "Tasso di Rilevamento OSR (@10% FPR)"
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/locales/en.json frontend/src/locales/it.json
git commit -m "feat: i18n keys for experiments matrix"
```

---

### Task 5: Build finale e PR

- [ ] **Step 1: Build completo**

```bash
cd frontend && npm run build
```

Nessun errore.

- [ ] **Step 2: Push e CI**

```bash
git push origin main
```

CI esegue test → docker → build frontend. Se passa, deploy in VPS.

---

### Riepilogo file per task

| Task | File | Modifica |
|------|------|----------|
| 1 | `frontend/src/features/explorer/types.ts` | Helper family/series/label + costanti |
| 2 | `frontend/src/features/explorer/components/BenchmarkMatrix.tsx` | **Nuovo** |
| 3 | `frontend/src/pages/ExperimentsPage.tsx` | Riscritto |
| 4 | `frontend/src/locales/en.json` | Nuove chiavi experiments.filters + matrix |
| 4 | `frontend/src/locales/it.json` | Nuove chiavi experiments.filters + matrix |
