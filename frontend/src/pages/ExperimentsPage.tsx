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
import type { ExperimentSummary } from "@/lib/types";
import { formatPercent } from "@/lib/format";

const SORTABLE_COLUMNS = [
  { key: "id", labelKey: "experiments.table.experiment" },
  { key: "model_variant", labelKey: "experiments.table.model" },
  { key: "mean_f1", labelKey: "experiments.table.f1", align: "right" },
  { key: "test_f1", labelKey: "experiments.table.test_f1", align: "right" },
  { key: "test_accuracy", labelKey: "experiments.table.test_acc", align: "right" },
  { key: "ncd_auroc", labelKey: "experiments.table.ncd_auroc", align: "right" },
] as const;

type Col = typeof SORTABLE_COLUMNS[number];
type SortKey = Col["key"];

function colLabel(col: Col, t: (k: string) => string): string {
  return t(col.labelKey);
}

function colAlign(col: Col): "left" | "right" | undefined {
  return "align" in col ? col.align : undefined;
}

const ALL_FAMILIES = ["ResNet", "ConvNeXt", "DINOv2"] as const;
// Dinamico: mostra solo le serie effettivamente presenti nei dati (supporta nuove serie senza hardcode)
const KNOWN_SERIES = ["Studio I-A", "Studio I-B", "Studio I-C", "Studio II-A", "Studio II-B"] as const;

export function ExperimentsPage() {
  const { t } = useTranslation();
  const { data: experiments, isLoading, isError, error, refetch } = useExperiments();

  const [view, setView] = useState<"matrix" | "table">("matrix");
  const [familyFilter, setFamilyFilter] = useState<string | null>(null);
  const [seriesFilter, setSeriesFilter] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const allSeries = useMemo(() => {
    if (!experiments) return [...KNOWN_SERIES] as string[];
    const seen = new Set(experiments.filter((e) => !e.id.includes("exp10")).map((e) => getExperimentSeries(e.id)).filter(Boolean) as string[]);
    // Mantieni ordine KNOWN_SERIES + eventuali nuove serie in coda ordinata
    const extra = [...seen].filter((s) => !(KNOWN_SERIES as readonly string[]).includes(s)).sort();
    return [...KNOWN_SERIES.filter((s) => seen.has(s)), ...extra] as string[];
  }, [experiments]);

  const filtered = useMemo(() => {
    if (!experiments) return [];
    return experiments.filter((e) => {
      if (e.id.includes("exp10")) return false;
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

  const { key, dir, toggle, sorted } = useSort<ExperimentSummary, SortKey>(filtered, "mean_f1", "desc");

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
            {allSeries.map((s) => <option key={s} value={s}>{s}</option>)}
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
          <Button variant="outline" size="sm" className="mt-4" onClick={() => refetch()}>
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
                  {SORTABLE_COLUMNS.map((col) => (
                    <SortHeader
                      key={col.key}
                      label={colLabel(col, t)}
                      sortKey={col.key}
                      activeKey={key}
                      dir={dir}
                      onToggle={toggle}
                      align={colAlign(col)}
                    />
                  ))}
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
                        {exp.mean_f1 != null ? `${formatPercent(exp.mean_f1, 2)}` : "—"}
                        {exp.std_f1 != null && (
                          <span className="text-muted-foreground text-xs ml-1">
                            ±{formatPercent(exp.std_f1, 2)}
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm tabular-nums">
                        {exp.test_f1 != null ? `${formatPercent(exp.test_f1, 2)}` : "—"}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm tabular-nums">
                        {exp.test_accuracy != null ? `${formatPercent(exp.test_accuracy, 2)}` : "—"}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm tabular-nums">
                        {exp.ncd_auroc != null ? `${formatPercent(exp.ncd_auroc, 2)}` : "—"}
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
