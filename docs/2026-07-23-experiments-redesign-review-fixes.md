# Experiments Page — Code Review Fixes

> Engineering spec per risolvere **tutti** i bug e il tech debt individuati nella code review delle modifiche Experiments Page.
> Data: 2026-07-23 | Priorità: Alta

---

## Indice

1. [Panoramica](#1-panoramica)
2. [Bug trovati](#2-bug-trovati)
3. [Soluzioni architetturali](#3-soluzioni-architetturali)
4. [File plan](#4-file-plan)
5. [Test](#5-test)

---

## 1. Panoramica

La code review della Experiments Page redesign ha evidenziato 17 problemi: 4 critici e 13 medi. Questa spec definisce le correzioni con scelte a lungo termine: componenti riutilizzabili, nessun type detection runtime, piena keyboard accessibility, DOM valido, e test per la logica pura.

---

## 2. Bug trovati

| # | File | Line | Problema | Gravità |
|---|------|------|----------|---------|
| 1 | `BenchmarkMatrix.tsx` | 97 | `\n` in `title` HTML — non cross-browser | 🔴 Critical |
| 2 | `BenchmarkMatrix.tsx` | 88-91 | `Math.max` ricalcolato per ogni cella | 🟡 Medium |
| 3 | `BenchmarkMatrix.tsx` | 45-49 | `labelToVariant` ricreato a ogni render | 🟡 Medium |
| 4 | `BenchmarkMatrix.tsx` | 150-155 | Duplicazione row rendering F1/OSR | 🟡 Medium |
| 5 | `BenchmarkMatrix.tsx` | 103 | `hover:scale-105` overflow nello scroll container | 🟡 Medium |
| 6 | `BenchmarkMatrix.tsx` | 19-22 | AUROC usa stessa scala colore di F1 | 🟡 Medium |
| 7 | `BenchmarkMatrix.tsx` | 29-36 | `lookup` non gestisce duplicati: stesso variant+series sovrascrive silenziosamente | 🔴 Critical |
| 8 | `BenchmarkMatrix.tsx` | — | Nessuna keyboard navigation — click-only | 🔴 Critical |
| 9 | `ExperimentsPage.tsx` | 52-56 | Sort comparator si rompe con campi null | 🔴 Critical |
| 10 | `ExperimentsPage.tsx` | 145 | Disabled checkbox senza onChange + nessun label accessibile | 🟡 Medium |
| 11 | `ExperimentsPage.tsx` | 128-137 | Compare button in matrix view: selected vuoto per sempre | 🟡 Medium |
| 12 | `ExperimentsPage.tsx` | 90,93 | Bottoni "Matrix"/"Table" hardcoded | 🟡 Medium |
| 13 | `ExperimentsPage.tsx` | 204 | `num_folds` null → `"5/null"` | 🟡 Medium |
| 14 | `ExperimentsPage.tsx` | — | Template string invece di `cn()` | 🟡 Medium |
| 15 | `sortByKey` (spec) | — | Type detection runtime fragile | 🟡 Medium |
| 16 | `MatrixSection` (spec) | — | `bestPerColumn` calcolato dentro lookup non suo | 🟡 Medium |
| 17 | `src/lib` (spec) | — | Nessun test per `compareNullSafe` | 🟡 Medium |

---

## 3. Soluzioni architetturali

### 3.1 `compareNullSafe` — `src/lib/sort.ts`

Funzione pura. Nessun type detection runtime. I tipi sono risolti dal chiamante TypeScript.

```typescript
export type SortDir = "asc" | "desc";
type SortablePrimitive = string | number | boolean | null | undefined;

export function compareNullSafe<T extends SortablePrimitive>(
  a: T,
  b: T,
  dir: SortDir = "asc",
): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (a < b) return dir === "asc" ? -1 : 1;
  if (a > b) return dir === "asc" ? 1 : -1;
  return 0;
}
```

Design rationale:
- `T extends SortablePrimitive` — TypeScript rifiuta oggetti o array in compile-time
- `<` e `>` funzionano per number, string, boolean
- Nessun `typeof`, nessun cast
- `== null` cattura sia `null` che `undefined`
- **Sempre null-last** indipendentemente dalla direzione (lo standard UI delle tabelle)

### 3.2 `useSort` — `src/hooks/useSort.ts`

Hook con `useReducer` per evitare side effect dentro updater:

```typescript
import { useReducer, useMemo, useCallback } from "react";
import type { SortDir } from "@/lib/sort";
import { compareNullSafe } from "@/lib/sort";

interface SortState<K> {
  key: K;
  dir: SortDir;
}

type SortAction<K> = { key: K };

function sortReducer<K>(state: SortState<K>, action: SortAction<K>): SortState<K> {
  if (action.key === state.key) {
    return { key: action.key, dir: state.dir === "asc" ? "desc" : "asc" };
  }
  return { key: action.key, dir: "asc" };
}

export function useSort<T, K extends keyof T>(items: T[], defaultKey: K) {
  const [state, dispatch] = useReducer(sortReducer<K>, {
    key: defaultKey,
    dir: "asc",
  });

  const sorted = useMemo(() => {
    const copy = [...items];
    copy.sort((a, b) => compareNullSafe(a[state.key], b[state.key], state.dir));
    return copy;
  }, [items, state.key, state.dir]);

  const toggle = useCallback((key: string) => dispatch({ key: key as unknown as K }), []);

  return {
    key: state.key as K,
    dir: state.dir,
    toggle,
    sorted,
  };
}
```

Design rationale:
- `useReducer` elimina il side effect dentro updater
- `toggle` accetta `string` (non `K`) per evitare problemi di inferenza TypeScript quando usato con `SortHeader` multi-colonna — il cast `as unknown as K` è sicuro perché i chiamanti passano solo chiavi valide
- `toggle` è stabile (`useCallback` con `[]`, referenza costante) — nessuna re-render chain inutile
- `sorted` è O(n log n) computed su ogni cambio di key/dir o items

### 3.3 `SortHeader` — `src/components/ui/SortHeader.tsx`

```typescript
import { TableHead } from "@/components/ui/table";
import type { SortDir } from "@/lib/sort";
import { cn } from "@/lib/utils";

interface SortHeaderProps {
  label: string;
  sortKey: string;
  activeKey: string;
  dir: SortDir;
  onToggle: (key: string) => void;
  align?: "left" | "right";
  className?: string;
}

export function SortHeader({
  label, sortKey, activeKey, dir, onToggle, align = "left", className,
}: SortHeaderProps) {
  const isActive = sortKey === activeKey;
  return (
    <TableHead
      className={cn(
        "cursor-pointer select-none",
        align === "right" && "text-right",
        className,
      )}
      onClick={() => onToggle(sortKey)}
      aria-sort={isActive ? (dir === "asc" ? "ascending" : "descending") : undefined}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {isActive && (
          <span className="text-xs text-muted-foreground" aria-hidden="true">
            {dir === "asc" ? "↑" : "↓"}
          </span>
        )}
      </span>
    </TableHead>
  );
}
```

Note:
- `aria-sort` è `undefined` (non `"none"`) quando inattivo — l'attributo non viene renderizzato
- `TableHead` importato da `@/components/ui/table`
- Parametri chiave come `string` invece di generics per evitare problemi di inferenza con toggle multi-colonna`

### 3.4 Tooltip inline — `<td>` con CSS `group`

Il componente shadcn `Tooltip` wrappa in `<button>`, invalido dentro `<tr>`. Un wrapper `<div>` attorno al `<td>` è igualmente invalido (il `<td>` deve essere figlio diretto di `<tr>`).

**Soluzione**: tooltip inline DENTRO il `<td>`, usando `group` sul `<td>` stesso:

```tsx
<td
  className={cn(
    "group relative text-center py-2 px-3 rounded-lg transition-all",
    exp
      ? "cursor-pointer focus-visible:ring-2 focus-visible:ring-ring hover:brightness-110"
      : "text-muted-foreground",
  )}
  style={{ backgroundColor: value !== null ? colorFn(value) : undefined }}
  onClick={() => exp && navigate(`/experiments/${exp.id}`)}
  onKeyDown={(e) => {
    if ((e.key === "Enter" || e.key === " ") && exp) {
      e.preventDefault();
      navigate(`/experiments/${exp.id}`);
    }
  }}
  tabIndex={exp ? 0 : undefined}
  role={exp ? "button" : undefined}
  aria-label={exp ? `View details: ${exp.name}` : undefined}
>
  <span>{formattedValue}</span>
  {isBest && <span className="ml-0.5 text-yellow-300" aria-label="best">★</span>}

  {/* Tooltip — DOM valido: div dentro td */}
  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block group-focus-within:block z-50">
    <div className="bg-popover text-popover-foreground text-xs rounded-lg border border-border px-3 py-2 shadow-md whitespace-nowrap">
      <p className="font-semibold">{exp?.name}</p>
      <p>{tooltipLabel}: {formattedValue}%</p>
      {value !== null && exp?.std_f1 != null && (
        <p>Std: ±{(exp.std_f1 * 100).toFixed(2)}%</p>
      )}
      {exp?.test_accuracy != null && (
        <p>Test Acc: {(exp.test_accuracy * 100).toFixed(2)}%</p>
      )}
    </div>
  </div>
</td>
```

Nota: nessun `pointer-events-none` — il tooltip cattura eventi naturalmente, l'hover sul tooltip mantiene `:hover` sul `<td>` evitando flickering quando l'utente sposta il mouse per leggere il tooltip.

Nessun wrapper, nessun `<button>`, nessun `<div>` tra `<tr>` e `<td>`. Pienamente valido HTML.

### 3.5 `MatrixSection` — `src/features/explorer/components/MatrixSection.tsx`

```typescript
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import type { ExperimentSummary } from "@/lib/types";
import {
  getArchitectureFamily, FAMILY_COLORS, LABEL_TO_VARIANT,
} from "@/features/explorer/types";

interface MatrixSectionProps {
  lookup: Record<string, Record<string, ExperimentSummary>>;
  architectureRows: string[];
  columns: string[];
  title?: string;
  accessor: (exp: ExperimentSummary) => number | null;
  colorFn: (value: number) => string;
  bestPerColumn: Record<string, number>;
  tooltipLabel: string;
}

export function MatrixSection({
  lookup, architectureRows, columns, title, accessor, colorFn, bestPerColumn, tooltipLabel,
}: MatrixSectionProps) {
  const navigate = useNavigate();
  const { t } = useTranslation();

  const renderCell = (col: string, variant: string) => {
    const exp = lookup[variant]?.[col];
    const value = exp ? accessor(exp) : null;
    const isBest = value !== null && value === bestPerColumn[col];
    const formattedValue = value !== null ? `${(value * 100).toFixed(1)}` : "—";

    return (
      <td
        key={col}
        className={cn(
          "group relative text-center py-2 px-3 rounded-lg transition-all",
          exp
            ? "cursor-pointer focus-visible:ring-2 focus-visible:ring-ring hover:brightness-110"
            : "text-muted-foreground",
        )}
        style={{ backgroundColor: value !== null ? colorFn(value) : undefined }}
        onClick={() => exp && navigate(`/experiments/${exp.id}`)}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && exp) {
            e.preventDefault();
            navigate(`/experiments/${exp.id}`);
          }
        }}
        tabIndex={exp ? 0 : undefined}
        role={exp ? "button" : undefined}
        aria-label={exp ? `View details: ${exp.name}` : undefined}
      >
        <span>{formattedValue}</span>
        {isBest && <span className="ml-0.5 text-yellow-300" aria-label="best performing architecture">★</span>}
        {exp && (
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block group-focus-within:block z-50">
            <div className="bg-popover text-popover-foreground text-xs rounded-lg border border-border px-3 py-2 shadow-md whitespace-nowrap">
              <p className="font-semibold">{exp.name}</p>
              <p>{tooltipLabel}: {formattedValue}%</p>
              {value !== null && exp.std_f1 != null && (
                <p>Std: ±{(exp.std_f1 * 100).toFixed(2)}%</p>
              )}
              {exp.test_accuracy != null && (
                <p>Test Acc: {(exp.test_accuracy * 100).toFixed(2)}%</p>
              )}
            </div>
          </div>
        )}
      </td>
    );
  };

  return (
    <div>
      {title && (
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          {title}
        </h3>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left py-2 pr-4 font-medium text-muted-foreground text-xs uppercase tracking-wider">
                {t("experiments.table.model")}
              </th>
              {columns.map((s) => (
                <th key={s} className="text-center py-2 px-3 font-medium text-xs uppercase tracking-wider text-muted-foreground">
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border/20">
            {architectureRows.map((archLabel) => {
              const variant = LABEL_TO_VARIANT[archLabel];
              const family = variant ? getArchitectureFamily(variant) : null;
              return (
                <tr key={archLabel} className="hover:bg-muted/20 transition-colors">
                  <td className="py-2 pr-4 font-medium text-sm flex items-center gap-2">
                    <span
                      className="inline-block size-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: family ? FAMILY_COLORS[family] : "gray" }}
                    />
                    <span>{archLabel}</span>
                  </td>
                  {columns.map((col) => renderCell(col, variant))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

Note:
- `renderCell` estrae la logica di cella: tooltip inline via `group` sul `<td>`, nessun wrapper
- `focus-visible:ring-2` per keyboard focus ring visibile
- `hover:brightness-110` invece di `hover:scale-105` — non causa overflow
- `role="button"` + `tabIndex={0}` + `onKeyDown` per keyboard accessibility
- Tooltip è un `<div>` dentro il `<td>` — DOM perfettamente valido
- Tooltip è un `<div>` dentro il `<td>` — DOM perfettamente valido, nessun `pointer-events-none` per evitare flickering hover
- `bestPerColumn` è una prop — il componente non accede a `lookup` per calcolarlo

### 3.6 `BenchmarkMatrix` refactored

```typescript
import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { ExperimentSummary } from "@/lib/types";
import {
  getExperimentSeries, getArchitectureFamily,
  ARCHITECTURE_ORDER, FAMILY_COLORS, LABEL_TO_VARIANT,
} from "@/features/explorer/types";
import { MatrixSection } from "./MatrixSection";

interface BenchmarkMatrixProps {
  experiments: ExperimentSummary[];
  familyFilter: string | null;
  seriesFilter: string | null;
}

const MATRIX_SERIES = ["Exp 6", "Exp 7", "Exp 8", "Exp 10"] as const;
const OSR_SERIES = ["Exp 8", "Exp 10"] as const;

function f1Color(f1: number): string {
  const l = 0.15 + f1 * 0.6;
  return `oklch(${l} 0.15 ${f1 > 0.6 ? 145 : 25})`;
}

function aurocColor(auroc: number): string {
  // 0.5 = random → grigio neutro; 1.0 = perfetto → verde
  const aboveRandom = Math.max(0, (auroc - 0.5) * 2);
  if (aboveRandom < 0.05) return "oklch(0.45 0.015 200)";
  const l = 0.35 + aboveRandom * 0.4;
  const chroma = 0.04 + aboveRandom * 0.14;
  return `oklch(${l} ${chroma} 145)`;
}

export function BenchmarkMatrix({ experiments, familyFilter, seriesFilter }: BenchmarkMatrixProps) {
  const { t } = useTranslation();

  const lookup = useMemo(() => {
    const map: Record<string, Record<string, ExperimentSummary>> = {};
    for (const exp of experiments) {
      const v = exp.model_variant;
      const s = getExperimentSeries(exp.id);
      if (!v || !s) continue;
      const existing = map[v]?.[s];
      if (existing) {
        // Tiebreaker: più folds completati, poi più recente, poi lexicographic id
        if (existing.folds_completed !== exp.folds_completed) {
          if ((existing.folds_completed ?? 0) > (exp.folds_completed ?? 0)) continue;
        } else if ((existing.timestamp ?? "") > (exp.timestamp ?? "")) continue;
        else if (existing.id < exp.id) continue;
      }
      if (!map[v]) map[v] = {};
      map[v][s] = exp;
    }
    return map;
  }, [experiments]);

  const visibleSeries = useMemo(() =>
    seriesFilter ? [seriesFilter] : [...MATRIX_SERIES],
  [seriesFilter]);

  const visibleOSR = useMemo(() =>
    seriesFilter
      ? (OSR_SERIES.includes(seriesFilter as typeof OSR_SERIES[number]) ? [seriesFilter] : [])
      : [...OSR_SERIES],
  [seriesFilter]);

  const architectureRows = useMemo(() =>
    ARCHITECTURE_ORDER.filter((archLabel) => {
      if (!familyFilter) return true;
      const variant = LABEL_TO_VARIANT[archLabel];
      const family = variant ? getArchitectureFamily(variant) : null;
      return family === familyFilter;
    }),
  [familyFilter]);

  const f1Best = useMemo(() => {
    const best: Record<string, number> = {};
    for (const col of visibleSeries) {
      best[col] = Math.max(
        ...architectureRows.map((arch) => {
          const exp = lookup[LABEL_TO_VARIANT[arch]]?.[col];
          return exp?.mean_f1 ?? 0;
        }),
      );
    }
    return best;
  }, [lookup, visibleSeries, architectureRows]);

  const osrBest = useMemo(() => {
    const best: Record<string, number> = {};
    for (const col of visibleOSR) {
      best[col] = Math.max(
        ...architectureRows.map((arch) => {
          const exp = lookup[LABEL_TO_VARIANT[arch]]?.[col];
          return exp?.ncd_auroc ?? 0;
        }),
      );
    }
    return best;
  }, [lookup, visibleOSR, architectureRows]);

  return (
    <div className="space-y-8">
      <MatrixSection
        lookup={lookup}
        architectureRows={architectureRows}
        columns={visibleSeries}
        accessor={(e) => e.mean_f1}
        colorFn={f1Color}
        bestPerColumn={f1Best}
        tooltipLabel="F1 CV"
      />
      {visibleOSR.length > 0 && (
        <MatrixSection
          lookup={lookup}
          architectureRows={architectureRows}
          columns={visibleOSR}
          title={t("experiments.matrix.osr_title")}
          accessor={(e) => e.ncd_auroc}
          colorFn={aurocColor}
          bestPerColumn={osrBest}
          tooltipLabel="AUROC"
        />
      )}
    </div>
  );
}
```

Tiebreaker rationale:
- Primo criterio: `folds_completed` (più alto vince — esperimento più completo)
- Secondo criterio: `timestamp` (più recente vince)
- Terzo criterio: `id` lexicographic (deterministico — sempre lo stesso risultato)

`visibleSeries`, `visibleOSR`, e `architectureRows` sono memoizzati con `useMemo` per garantire stabilità referenziale. Senza, i `useMemo` di `f1Best` e `osrBest` riceverebbero nuove referenze a ogni render e non farebbero mai caching.

Entrambi i `useMemo` di best-per-column sono O(architetture × colonne) = O(6 × 4) = 24 iterazioni.

### 3.7 `ExperimentsPage.tsx` fixes

```typescript
import { useState, useMemo, useCallback } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { FlaskConical, Table2, Grid3X3, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
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
import { useSort } from "@/hooks/useSort";
import { SortHeader } from "@/components/ui/SortHeader";

type SortKey = "id" | "model_variant" | "mean_f1" | "test_accuracy" | "ncd_auroc";

const ALL_FAMILIES = ["ResNet", "ConvNeXt", "DINOv2"] as const;
const ALL_SERIES = ["Exp 6", "Exp 7", "Exp 8", "Exp 10"] as const;

export function ExperimentsPage() {
  const { t } = useTranslation();
  const { data: experiments, isLoading, isError, error } = useExperiments();

  const [view, setView] = useState<"matrix" | "table">("matrix");
  const [familyFilter, setFamilyFilter] = useState<string | null>(null);
  const [seriesFilter, setSeriesFilter] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

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

  const { key, dir, toggle, sorted } = useSort(filtered, "id");

  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < 4) next.add(id);
      return next;
    });
  }, []);

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700">
      <div className="flex items-center gap-4">
        <div className="size-10 md:size-12 rounded-2xl bg-purple-500/10 flex items-center justify-center shrink-0 text-purple-500">
          <FlaskConical className="size-6" />
        </div>
        <div>
          <h1 className="text-2xl md:text-4xl font-black tracking-tight">{t("experiments.title")}</h1>
          <p className="text-muted-foreground text-sm mt-1">{t("experiments.subtitle")}</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center justify-between">
        <div className="flex gap-2">
          <Button variant={view === "matrix" ? "default" : "outline"} size="sm" onClick={() => setView("matrix")} aria-pressed={view === "matrix"}>
            <Grid3X3 className="size-4 mr-1" /> {t("experiments.view_matrix")}
          </Button>
          <Button variant={view === "table" ? "default" : "outline"} size="sm" onClick={() => setView("table")} aria-pressed={view === "table"}>
            <Table2 className="size-4 mr-1" /> {t("experiments.view_table")}
          </Button>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <select value={familyFilter ?? ""} onChange={(e) => setFamilyFilter(e.target.value || null)}
            className="text-xs bg-card border border-border/50 rounded-lg px-2.5 py-1.5 text-foreground cursor-pointer appearance-auto">
            <option value="">{t("experiments.filters.all_architectures")}</option>
            {ALL_FAMILIES.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          <select value={seriesFilter ?? ""} onChange={(e) => setSeriesFilter(e.target.value || null)}
            className="text-xs bg-card border border-border/50 rounded-lg px-2.5 py-1.5 text-foreground cursor-pointer appearance-auto">
            <option value="">{t("experiments.filters.all_experiments")}</option>
            {ALL_SERIES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      ) : isError ? (
        <div className="text-center py-20 text-muted-foreground">
          <AlertCircle className="size-12 mx-auto mb-4 opacity-30 text-destructive" />
          <p className="text-destructive font-medium">{t("common.error")}</p>
          {error instanceof Error && (
            <p className="text-xs mt-2 text-muted-foreground max-w-md mx-auto">{error.message}</p>
          )}
          <Button variant="outline" size="sm" className="mt-4" onClick={() => window.location.reload()}>
            {t("common.retry")}
          </Button>
        </div>
      ) : !experiments || experiments.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">
          <FlaskConical className="size-12 mx-auto mb-4 opacity-30" />
          <p>{t("experiments.empty")}</p>
        </div>
      ) : view === "matrix" ? (
        <BenchmarkMatrix
          experiments={experiments}
          familyFilter={familyFilter}
          seriesFilter={seriesFilter}
        />
      ) : (
        <>
          <div className="rounded-lg border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10">
                    <div className="size-4" aria-hidden="true" />
                    <span className="sr-only">{t("experiments.table.select")}</span>
                  </TableHead>
                  <SortHeader label={t("experiments.table.experiment")} sortKey="id" activeKey={key} dir={dir} onToggle={toggle} />
                  <SortHeader label={t("experiments.table.model")} sortKey="model_variant" activeKey={key} dir={dir} onToggle={toggle} />
                  <SortHeader label="F1" sortKey="mean_f1" activeKey={key} dir={dir} onToggle={toggle} align="right" />
                  <SortHeader label={t("experiments.table.test_acc")} sortKey="test_accuracy" activeKey={key} dir={dir} onToggle={toggle} align="right" />
                  <SortHeader label="NCD AUROC" sortKey="ncd_auroc" activeKey={key} dir={dir} onToggle={toggle} align="right" />
                  <TableHead className="text-right">{t("experiments.table.folds")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.map((exp) => {
                  const family = getArchitectureFamily(exp.model_variant);
                  const series = getExperimentSeries(exp.id);
                  return (
                    <TableRow key={exp.id} className={cn(selected.has(exp.id) && "bg-primary/5")}>
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
                        {exp.folds_completed ?? "—"}/{exp.num_folds ?? "—"}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
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

Changes rispetto alla versione attuale:
- `cn()` importato e usato per classi condizionali su `TableRow`
- `useSort(filtered, "id")` sostituisce il sort manuale
- `SortHeader` sostituisce i `<TableHead>` con `onClick` manuali
- Bottoni view usano `t("experiments.view_matrix")` e `t("experiments.view_table")`
- Header checkbox → `<div>` decorativo + `<span className="sr-only">`
- No compare button in matrix view
- `num_folds` → `exp.num_folds ?? "—"`

### 3.8 `types.ts` — `LABEL_TO_VARIANT`

```typescript
export const LABEL_TO_VARIANT: Readonly<Record<string, string>> = {
  "ResNet-18": "resnet18", "ResNet-50": "resnet50",
  "ConvNeXt-Tiny": "convnext_tiny", "ConvNeXt-Small": "convnext_small",
  "DINOv2-Small": "dinov2_small", "DINOv2-Base": "dinov2_base",
};
```

`Readonly<>` impedisce mutazioni accidentali runtime.

### 3.9 Locale keys

EN:
```json
{
  "experiments": {
    "view_matrix": "Matrix",
    "view_table": "Table",
    "table": {
      "select": "Select experiments to compare"
    }
  }
}
```

IT:
```json
{
  "experiments": {
    "view_matrix": "Matrix",
    "view_table": "Tabella",
    "table": {
      "select": "Seleziona esperimenti da confrontare"
    }
  }
}
```

---

## 4. File plan

Prima di iniziare: `mkdir -p src/lib/__tests__ src/hooks src/components/ui src/features/explorer/components`

```bash
npm install -D vitest
```

| Ordine | File | Modifica |
|--------|------|----------|
| 0 | `mkdir -p src/lib/__tests__` | Creare directory per i test |
| 0b | `npm install -D vitest` | Installare test runner |
| 1 | `src/lib/sort.ts` | **Nuovo** — `compareNullSafe` + `SortDir` export |
| 2 | `src/lib/__tests__/sort.test.ts` | **Nuovo** — test esaustivi |
| 3 | `src/hooks/useSort.ts` | **Nuovo** — hook con `useReducer` |
| 4 | `src/components/ui/SortHeader.tsx` | **Nuovo** — SortHeader con `aria-sort` |
| 5 | `src/features/explorer/types.ts` | Aggiungere `LABEL_TO_VARIANT` |
| 6 | `src/features/explorer/components/MatrixSection.tsx` | **Nuovo** — table renderer unico con tooltip inline |
| 7 | `src/features/explorer/components/BenchmarkMatrix.tsx` | Refactor completo |
| 8 | `src/pages/ExperimentsPage.tsx` | Refactor: useSort, SortHeader, cn(), a11y, i18n |
| 9 | `frontend/package.json` | Aggiungere `"test": "vitest run"` agli scripts |
| 10 | `src/locales/en.json` | + view_matrix, view_table, table.select |
| 11 | `src/locales/it.json` | + view_matrix, view_table, table.select |

**Dipendenze**:
- 2 → 1, 0b (test dipende dalla funzione + vitest)
- 3 → 1 (useSort dipende da compareNullSafe + SortDir)
- 4 → 1 (SortHeader dipende da SortDir), `TableHead` da `@/components/ui/table`
- 6 → 5 (MatrixSection dipende da types)
- 7 → 5, 6 (BenchmarkMatrix dipende da types + MatrixSection)
- 8 → 3, 4, 9 (ExperimentsPage dipende da useSort + SortHeader)

---

## 5. Test

### `src/lib/__tests__/sort.test.ts`

```typescript
import { describe, it, expect } from "vitest";
import { compareNullSafe } from "../sort";

describe("compareNullSafe", () => {
  describe("numbers", () => {
    it("asc: a < b returns -1", () => { expect(compareNullSafe(1, 2, "asc")).toBe(-1); });
    it("asc: a > b returns 1", () => { expect(compareNullSafe(2, 1, "asc")).toBe(1); });
    it("asc: a === b returns 0", () => { expect(compareNullSafe(1, 1, "asc")).toBe(0); });
    it("desc: a < b returns 1", () => { expect(compareNullSafe(1, 2, "desc")).toBe(1); });
    it("desc: a > b returns -1", () => { expect(compareNullSafe(2, 1, "desc")).toBe(-1); });
    it("desc: a === b returns 0", () => { expect(compareNullSafe(1, 1, "desc")).toBe(0); });
  });

  describe("strings", () => {
    it("asc: a < b returns negative", () => { expect(compareNullSafe("a", "b", "asc")).toBeLessThan(0); });
    it("asc: a > b returns positive", () => { expect(compareNullSafe("b", "a", "asc")).toBeGreaterThan(0); });
    it("desc: a < b returns positive", () => { expect(compareNullSafe("a", "b", "desc")).toBeGreaterThan(0); });
  });

  describe("null and undefined", () => {
    it("null vs value: null last asc", () => { expect(compareNullSafe(null, 1, "asc")).toBe(1); });
    it("value vs null: value first asc", () => { expect(compareNullSafe(1, null, "asc")).toBe(-1); });
    it("null vs value: null last desc", () => { expect(compareNullSafe(null, 1, "desc")).toBe(1); });
    it("value vs null: value first desc", () => { expect(compareNullSafe(1, null, "desc")).toBe(-1); });
    it("null vs null returns 0", () => { expect(compareNullSafe(null, null, "asc")).toBe(0); });
    it("undefined treated as null", () => { expect(compareNullSafe(undefined, 1, "asc")).toBe(1); });
    it("both undefined returns 0", () => { expect(compareNullSafe(undefined, undefined, "asc")).toBe(0); });
  });

  describe("default direction", () => {
    it("defaults to asc", () => { expect(compareNullSafe(1, 2)).toBe(-1); });
  });
});
```

---

## Appendice: Problemi risolti

| # | Problema | Soluzione | Stato |
|---|----------|-----------|-------|
| 1 | `\n` in title | Tooltip inline via `group` CSS dentro `<td>` | ✅ Risolto |
| 2 | `Math.max` per cella | `bestPerColumn` precalcolato in useMemo | ✅ Risolto |
| 3 | `labelToVariant` inline | `LABEL_TO_VARIANT` module-level const | ✅ Risolto |
| 4 | Duplicazione F1/OSR | `MatrixSection` component | ✅ Risolto |
| 5 | `scale` overflow | `brightness-110` | ✅ Risolto |
| 6 | AUROC scala F1 | `aurocColor()` separata con threshold 0.5 | ✅ Risolto |
| 7 | lookup sovrascrive | Tiebreaker folds → timestamp → id | ✅ Risolto |
| 8 | No keyboard nav | `tabIndex`, `onKeyDown`, `role="button"`, `focus-visible:ring` | ✅ Risolto |
| 9 | Sort su null | `compareNullSafe` — nessun type detection | ✅ Risolto |
| 10 | Disabled checkbox | `<div>` decorativo + sr-only label | ✅ Risolto |
| 11 | Compare dead code | Rimosso da matrix view | ✅ Risolto |
| 12 | Bottoni hardcoded | i18n `view_matrix`/`view_table` | ✅ Risolto |
| 13 | `num_folds` null | `?? "—"` | ✅ Risolto |
| 14 | No `cn()` | `import { cn }` + usato per `TableRow` | ✅ Risolto |
| 15 | Type detection sort | `<` e `>` nativi, nessun typeof | ✅ Risolto |
| 16 | `bestPerColumn` leak | Calcolato in BenchmarkMatrix, passato come prop | ✅ Risolto |
| 17 | Nessun test | `sort.test.ts` con 16 test case | ✅ Risolto |
