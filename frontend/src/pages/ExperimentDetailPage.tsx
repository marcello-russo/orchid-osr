import * as React from "react";
import { useState, useCallback } from "react";
import { useParams, Link, useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ChevronLeft, ChevronRight, GitCompare } from "lucide-react";
import { fetchNcdResults, fetchExperimentLog } from "@/lib/api";
import { AblationChart, ClusteringComparisonChart, PerClassAurocChart } from "@/features/explorer/components/NcdCharts";
import { Scatter2D } from "@/features/embeddings/Scatter2D";
import { Scatter3D } from "@/features/embeddings/Scatter3D";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useExperiment } from "@/features/explorer/hooks/useExperiment";
import type { NcdClusteringResult } from "@/lib/types";
import { FoldSelector } from "@/features/explorer/components/FoldSelector";
import { LossCurvesChart } from "@/features/explorer/components/LossCurvesChart";
import { MetricCurvesChart } from "@/features/explorer/components/MetricCurvesChart";
import { FoldSummaryChart } from "@/features/explorer/components/FoldSummaryChart";
import { ConfusionMatrixChart } from "@/features/explorer/components/ConfusionMatrixChart";
import { OverfittingProxyChart } from "@/features/explorer/components/OverfittingProxyChart";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import { ExperimentConfigCard } from "@/features/explorer/components/ExperimentConfigCard";
import { getTooltips } from "@/features/explorer/tooltipText";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useCompareExperiments } from "@/features/explorer/hooks/useCompareExperiments";
import { CompareModeView } from "@/features/explorer/components/CompareModeView";
import { ExperimentSelectTable } from "@/features/explorer/components/ExperimentSelectTable";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { useRef } from "react";
import { useChartSize } from "@/features/explorer/hooks/useChartSize";
import { useChartColors } from "@/features/explorer/hooks/useChartColors";

export function ExperimentDetailPage() {
  const { t } = useTranslation();
  const TOOLTIPS = getTooltips(t);
  const { id } = useParams<{ id: string }>();
  const expNum = id ? parseInt(id.match(/exp(\d+)/)?.[1] ?? "", 10) : NaN;
  const { data: experiment, isLoading, error } = useExperiment(id ?? "");
  const [searchParams, setSearchParams] = useSearchParams();
  const compareRaw = searchParams.get("compare") ?? "";
  const compareIds = compareRaw ? compareRaw.split(",").filter((x) => x && x !== id) : [];
  const isCompareMode = compareIds.length >= 1;
  const { experiments: compareExperiments, isLoading: compareLoading } = useCompareExperiments(
    isCompareMode ? compareIds : []
  );
  const [selectedFold, setSelectedFold] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState("charts");

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-96" />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Skeleton className="h-[350px] rounded-lg" />
          <Skeleton className="h-[350px] rounded-lg" />
        </div>
      </div>
    );
  }

  if (error || !experiment) {
    return (
      <div className="py-20 text-center text-muted-foreground">
        <p>{t("common.error")}</p>
        <Link to="/experiments" className="mt-2 inline-block text-primary hover:underline">
          {t("experiment_detail.back_to_experiments")}
        </Link>
      </div>
    );
  }

  const { config, folds, results, summary } = experiment;
  const hasFolds = folds.length > 0;
  const hasResults = results !== null;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div>
        <Link
          to="/experiments"
          className="mb-2 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-primary transition-colors duration-150"
        >
          <ArrowLeft className="h-4 w-4" />
          {t("common.experiments")}
        </Link>
        <div className="flex flex-wrap items-center gap-2">
          {isCompareMode ? (
            <>
              <span className="text-lg sm:text-2xl font-bold">{t("experiment.comparing")}</span>
              {[experiment, ...compareExperiments].filter(Boolean).map((exp, i) => {
                const COLORS = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6"];
                return (
                  <Badge key={exp.id} variant="outline" className="font-mono text-xs"
                    style={{ borderColor: COLORS[i % 4], color: COLORS[i % 4] }}>
                    {exp.id}
                  </Badge>
                );
              })}
            </>
          ) : (
            <>
              <h1 className="text-lg sm:text-2xl font-bold text-primary break-all">{id}</h1>
              <Badge variant="secondary" className="text-xs">
                {summary.type}
              </Badge>
              {summary.model_variant && (
                <Badge variant="outline" className="font-mono text-xs">
                  {summary.model_variant}
                </Badge>
              )}
            </>
          )}
        </div>
        <div className="mt-2 flex items-center gap-2">
          {!isCompareMode ? (
            <Dialog>
              <DialogTrigger render={<Button variant="outline" size="sm" />}>
                <GitCompare className="mr-1.5 size-4" /> Compare
              </DialogTrigger>
              <DialogContent className="sm:max-w-2xl md:max-w-3xl lg:max-w-4xl">
                <DialogHeader>
                  <DialogTitle>{t("experiment.select_compare")}</DialogTitle>
                </DialogHeader>
                <ExperimentSelectTable
                  currentId={id!}
                  onCompare={(selected) => setSearchParams({ compare: selected.join(",") })}
                />
              </DialogContent>
            </Dialog>
          ) : (
            <Button variant="ghost" size="sm" onClick={() => setSearchParams({})}>
              {t("experiment.exit_compare")}
            </Button>
          )}
        </div>

        {/* Config summary — only in single-experiment mode */}
        {!isCompareMode && (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted-foreground">

          {summary.test_accuracy !== null && (
            <span className="font-medium text-primary">
              {t("experiment_detail.metrics.accuracy")}: {(summary.test_accuracy * 100).toFixed(2)}%
              <InfoTooltip text={TOOLTIPS.testAccuracy} />
            </span>
          )}
          {summary.test_f1 !== null && (
            <span className="font-medium text-primary">
              {t("experiment_detail.metrics.f1")}: {(summary.test_f1 * 100).toFixed(2)}%
              <InfoTooltip text={TOOLTIPS.f1macro} />
            </span>
          )}
        </div>
        )}
      </div>

      {/* Training recipe card — replaced by CompareModeView in compare mode */}
      {isCompareMode && compareLoading ? (
        <Skeleton className="h-40 rounded-lg" />
      ) : isCompareMode ? (
        <CompareModeView
          experiments={[experiment, ...compareExperiments].filter(Boolean)}
        />
      ) : (
        <ExperimentConfigCard config={config} />
      )}

      {/* Fold selector — hidden in compare mode */}
      {hasFolds && !isCompareMode && (
        <FoldSelector
          folds={folds}
          selectedFold={selectedFold}
          onSelect={setSelectedFold}
        />
      )}

      {/* Tab bar — hidden in compare mode */}
      {!isCompareMode && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setActiveTab("charts")}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
              activeTab === "charts" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
            }`}
          >
            {t("experiment.charts")}
          </button>
          {(expNum === 8 || expNum === 9 || expNum === 10) && (
            <button
              onClick={() => setActiveTab("ncd")}
              className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
                activeTab === "ncd" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
              }`}
            >
              {t("experiment_detail.ncd_tab")}
            </button>
          )}
          <button
            onClick={() => setActiveTab("logs")}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
              activeTab === "logs" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
            }`}
          >
            {t("experiment.logs")}
          </button>
        </div>
      )}

      {/* Content — hidden in compare mode */}
      {!isCompareMode && activeTab === "charts" && (hasFolds ? (
        <>
        <ChartCarousel
          folds={folds}
          selectedFold={selectedFold}
          meanF1={summary.mean_f1}
          stdF1={summary.std_f1}
          confusionMatrix={
            hasResults && results.test_metrics
              ? { matrix: results.test_metrics.confusion_matrix, labels: results.test_metrics.confusion_matrix_labels }
              : null
          }
          experimentId={id}
        />
        </>
      ) : hasResults && results.test_metrics ? (
        <BaselineResults metrics={results.test_metrics} />
      ) : (
        <div className="py-10 text-center text-muted-foreground">
          {t("experiment_detail.no_results")}
        </div>
      ))}

      {!isCompareMode && activeTab === "ncd" && (expNum === 8 || expNum === 9 || expNum === 10) && (
        <NcdDetail modelVariant={summary.model_variant ?? config.model_variant ?? ""} exp={expNum} />
      )}

      {!isCompareMode && activeTab === "logs" && id && (
        <LogsPanel experimentId={id} />
      )}
    </div>
  );
}

/* ---- Chart Carousel ---- */

interface ChartCarouselProps {
  folds: import("@/lib/types").FoldData[];
  selectedFold: number | null;
  meanF1?: number | null;
  stdF1?: number | null;
  confusionMatrix: { matrix: number[][]; labels: string[] } | null;
  experimentId?: string;
}

function ChartCarousel({ folds, selectedFold, meanF1, stdF1, confusionMatrix, experimentId }: ChartCarouselProps) {
  const { t } = useTranslation();
  const hasPhases = folds.length > 0 && folds[0]?.fold_phases?.length ? folds[0].fold_phases.length > 0 : false;
  const hasOverfitting = folds.some(f => f.history?.val_loss?.some((v: number) => v > 0));
  const [active, setActive] = useState(0);

  /* ---------- Phase 1 (SupCon) helper ---------- */
  const COLORS = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6", "#EF4444"];
  const supconData = hasPhases ? folds[0]?.fold_phases?.[0] : null;
  const supconEpochs = supconData?.epochs ?? 0;
  const ceEpochs = folds.length > 0 ? folds[0].history.train_loss.length : 0;

  const supconChartData = React.useMemo(() => {
    if (!supconData) return [];
    return Array.from({ length: supconEpochs }, (_, ep) => {
      const row: Record<string, number | string> = { epoch: ep + 1 };
      for (const f of folds) {
        const m = f.fold_phases?.[0]?.metrics ?? {};
        const tl = m["train_loss"]?.[ep];
        const ka = m["val_knn_acc"]?.[ep];
        if (tl !== undefined) row[`tl_${f.fold}`] = tl;
        if (ka !== undefined) row[`ka_${f.fold}`] = ka;
      }
      return row;
    });
  }, [folds, supconData, supconEpochs]);

  /* ---------- Titles ---------- */
  const chart_titles = [
    ...(hasPhases ? [`SupCon Loss (${supconEpochs}ep)`, `KNN Accuracy (${supconEpochs}ep)`] : []),
    t("experiment_detail.charts.loss"),
    t("experiment_detail.charts.metrics"),
    t("experiment_detail.charts.folds"),
    ...(hasOverfitting ? [t("experiment_detail.charts.overfitting")] : []),
    ...(confusionMatrix ? [t("experiment_detail.charts.confusion")] : []),
  ];
  const total = chart_titles.length;
  const clamp = useCallback((i: number) => Math.max(0, Math.min(total - 1, i)), [total]);

  /* ---------- Chart renderer ---------- */
  const renderChart = (idx: number) => {
    let offset = 0;

    if (hasPhases) {
      if (idx === 0) {
        // SupCon Training Loss
        return (
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <span className="flex size-5 items-center justify-center rounded bg-indigo-500/10 text-indigo-500 text-[10px] font-bold">P1</span>
                <div>
                  <CardTitle className="text-sm">SupCon Training Loss</CardTitle>
                  <p className="text-[11px] text-muted-foreground mt-0.5">Phase 1: {supconEpochs} epochs | Phase 2 (CE): {ceEpochs} epochs | Total: {supconEpochs + ceEpochs}</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-2">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={supconChartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {folds.map((f, i) => (
                    <Line key={`tl_${f.fold}`} type="monotone" dataKey={`tl_${f.fold}`}
                      name={`Fold ${f.fold}`} stroke={COLORS[i % COLORS.length]}
                      dot={false} strokeWidth={1.5} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        );
      }
      if (idx === 1) {
        // KNN Validation Accuracy
        return (
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <span className="flex size-5 items-center justify-center rounded bg-indigo-500/10 text-indigo-500 text-[10px] font-bold">P1</span>
                <div>
                  <CardTitle className="text-sm">KNN Validation Accuracy</CardTitle>
                  <p className="text-[11px] text-muted-foreground mt-0.5">Phase 1: {supconEpochs} epochs | Phase 2 (CE): {ceEpochs} epochs | Total: {supconEpochs + ceEpochs}</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-2">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={supconChartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {folds.map((f, i) => (
                    <Line key={`ka_${f.fold}`} type="monotone" dataKey={`ka_${f.fold}`}
                      name={`Fold ${f.fold}`} stroke={COLORS[i % COLORS.length]}
                      dot={false} strokeWidth={1.5} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        );
      }
      offset = 2;
    }

    switch (idx - offset) {
      case 0: return <LossCurvesChart folds={folds} selectedFold={selectedFold} />;
      case 1: return <MetricCurvesChart folds={folds} selectedFold={selectedFold} />;
      case 2: return <FoldSummaryChart folds={folds} meanF1={meanF1} stdF1={stdF1} />;
      case 3: return hasOverfitting ? <OverfittingProxyChart folds={folds} selectedFold={selectedFold} /> : null;
      case 4: return confusionMatrix ? <ConfusionMatrixChart matrix={confusionMatrix.matrix} labels={confusionMatrix.labels} /> : null;
      default: return null;
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 overflow-x-auto scrollbar-none">
        <Button variant="ghost" size="icon-sm"
          onClick={() => setActive((a) => clamp(a - 1))}
          disabled={active === 0} className="shrink-0 hidden sm:flex"
        ><ChevronLeft className="size-4" /></Button>
        {Array.from({ length: total }, (_, i) => {
          const isPhase1 = hasPhases && i < 2;
          return (
            <button key={i} onClick={() => setActive(i)}
              className={cn(
                "shrink-0 rounded-full px-4 py-2 text-sm font-medium transition-colors whitespace-nowrap min-h-[44px] cursor-pointer",
                i === active ? "bg-primary text-primary-foreground" : "text-muted-foreground bg-muted/50 hover:bg-muted",
                isPhase1 && i !== active ? "text-indigo-500 bg-indigo-500/10 hover:bg-indigo-500/20" : ""
              )}
            >{chart_titles[i]}</button>
          );
        })}
        <Button variant="ghost" size="icon-sm"
          onClick={() => setActive((a) => clamp(a + 1))}
          disabled={active >= total - 1} className="shrink-0 hidden sm:flex"
        ><ChevronRight className="size-4" /></Button>
        {experimentId && (
          <Link to={`/embeddings?experiment=${experimentId}`}
            className="shrink-0 rounded-full px-4 py-2 text-sm font-medium transition-colors bg-indigo-500/10 text-indigo-500 hover:bg-indigo-500/20 border border-indigo-500/20"
          >🧬 {t("experiment_detail.latent_space")}</Link>
        )}
      </div>
      <div>{renderChart(active)}</div>
    </div>
  );
}

/* ---- Baseline Results ---- */

interface TestMetrics {
  accuracy: number;
  f1_macro: number;
  precision_macro?: number;
  recall_macro?: number;
  per_class_metrics?: Record<string, { precision: number; recall: number; f1: number; support: number }>;
}

function BaselineResults({ metrics }: { metrics: TestMetrics }) {
  const { t } = useTranslation();
  const chartRef = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(chartRef);
  const chartColors = useChartColors();

  const classData = Object.entries(metrics.per_class_metrics ?? {}).map(([className, c]) => ({
    name: className.replace("O. ", ""),
    F1: c.f1,
    Precision: c.precision,
    Recall: c.recall,
  }));

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label={t("experiment_detail.metrics.accuracy")} value={metrics.accuracy} />
        <MetricCard label={t("experiment_detail.metrics.f1")} value={metrics.f1_macro} />
        {metrics.precision_macro != null && <MetricCard label={t("experiment_detail.metrics.precision")} value={metrics.precision_macro} />}
        {metrics.recall_macro != null && <MetricCard label={t("experiment_detail.metrics.recall")} value={metrics.recall_macro} />}
      </div>

      {/* Per-class bar chart */}
      {classData.length > 0 && (
        <Card>
          <CardHeader className="pb-2 border-b border-border/50">
            <CardTitle className="text-base">{t("experiment_detail.per_class_metrics")}</CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            <div ref={chartRef} style={{ width: "100%", height }}>
              <BarChart data={classData} width={width} height={height}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
                <XAxis dataKey="name" tick={{ fontSize: 13 }} />
                <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 14 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(value) => (Number(value) * 100).toFixed(2) + "%"} />
                <Bar dataKey="F1" fill={chartColors[0]} radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
                <Bar dataKey="Precision" fill={chartColors[1]} radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
                <Bar dataKey="Recall" fill={chartColors[2]} radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
              </BarChart>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Per-class table */}
      {classData.length > 0 && (
        <Card>
          <CardHeader className="pb-2 border-b border-border/50">
            <CardTitle className="text-base">{t("experiment_detail.per_class_breakdown")}</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("experiment_detail.table.class")}</TableHead>
                  <TableHead className="text-right">{t("experiment_detail.metrics.precision")}</TableHead>
                  <TableHead className="text-right">{t("experiment_detail.metrics.recall")}</TableHead>
                  <TableHead className="text-right">{t("experiment_detail.charts.f1_label")}</TableHead>
                  <TableHead className="text-right whitespace-nowrap">{t("experiment_detail.table.support")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(metrics.per_class_metrics!).map(([className, c]) => (
                  <TableRow key={className}>
                    <TableCell className="font-medium">{className}</TableCell>
                    <TableCell className="text-right font-mono">{(c.precision * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right font-mono">{(c.recall * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right font-mono">{(c.f1 * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right whitespace-nowrap">{c.support}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

/* ---- NCD Detail ---- */

const _SPECIES_PALETTE: Record<string, string> = {
  "O. exaltata": "#3B82F6", "O. garganica": "#D97706", "O. incubacea": "#10B981",
  "O. majellensis": "#8B5CF6", "O. sphegodes": "#EF4444", "O. sphegodes_Palena": "#EC4899",
  "O. sphegodes_Napoli": "#14B8A6", "unknown": "#666666",
};

function NcdDetail({ modelVariant, exp }: { modelVariant: string; exp?: number }) {
  const { t } = useTranslation();
  const [tsneMode, setTsneMode] = useState<"2d" | "3d">("2d");
  const { data, isLoading, error } = useQuery({
    queryKey: ["ncd", modelVariant, exp ?? 8],
    queryFn: () => fetchNcdResults(modelVariant, exp ?? 8),
    staleTime: 5 * 60 * 1000,
    enabled: modelVariant.length > 0,
  });

  if (isLoading) return <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>;
  if (error || !data) return <p className="text-sm text-muted-foreground text-center py-8">{t("experiment_detail.ncd_no_data")}</p>;

  const detectors = data.detectors;
  const openset = data.openset;
  const clustering = data.clustering ?? {};
  const meanAuroc = detectors?.Ensemble?.auroc ?? openset?.mean_auroc ?? 0;
  const napoliAuroc = detectors?.Ensemble?.auroc ?? openset?.novel_class?.auroc;
  const ablation = openset?.ablation;
  const perClass = openset?.per_class_metrics ?? [];
  const tsne2d = data.tsne_2d as [number, number][] | undefined;
  const tsne3d = data.tsne_3d as [number, number, number][] | undefined;
  const labels = data.novel_class_label as number[] | undefined;

  // Compute best clustering ACC
  const bestClustering = Object.values(clustering).reduce<NcdClusteringResult | null>(
    (best, c) => (c.acc && c.acc > (best?.acc ?? 0) ? c : best), null
  );
  const silhouette = clustering.kmeans_best?.silhouette ?? null;

  // Build scatter points for t-SNE — map known classes to species names
  const classNames: string[] = data.class_names ?? [];
  const scatterPoints = tsne2d?.map((p, i) => ({
    x: p[0], y: p[1],
    species: labels?.[i] !== undefined && labels[i] >= 0 && labels[i] < classNames.length
      ? classNames[labels[i]] : "O. sphegodes_Napoli",
    idx: i,
  })) ?? [];

  const speciesSet = [...new Set(scatterPoints.map((p) => p.species))];
  const speciesColors = speciesSet.length > 0
    ? Object.fromEntries(speciesSet.map((s) => [s, _SPECIES_PALETTE[s] ?? "#666"]))
    : {};

  return (
    <div className="space-y-8 p-4">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-xl border border-border/40 bg-background p-4 text-center">
          <div className="text-xs text-muted-foreground uppercase font-bold mb-1">{t("experiment_detail.ncd_auroc")}</div>
          <div className={`text-2xl font-black ${meanAuroc < 0.5 ? "text-red-500" : "text-green-500"}`}>
            {(meanAuroc * 100).toFixed(1)}%
          </div>
          <div className="flex flex-wrap justify-center gap-1.5 mt-1.5">
            {["Ensemble", "Mahalanobis", "KNN", "Energy"].map((d) => {
              const val = detectors?.[d]?.auroc;
              return val != null ? (
                <span key={d} className="text-[10px] px-1.5 py-0.5 rounded bg-muted font-mono">
                  {d[0]}: {(val * 100).toFixed(1)}%
                </span>
              ) : null;
            })}
          </div>
        </div>
        {napoliAuroc !== undefined && (
          <div className="rounded-xl border-2 border-red-500/30 bg-red-500/5 p-4 text-center">
            <div className="text-xs text-muted-foreground uppercase font-bold mb-1">
              {t("experiment_detail.ncd_napoli_auroc")}
              <span className="ml-1.5 inline-block bg-red-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded">
                {t("experiment_detail.ncd_novel_label")}
              </span>
            </div>
            <div className="text-2xl font-black text-red-500">{(napoliAuroc * 100).toFixed(1)}%</div>
            <div className="text-[10px] text-muted-foreground mt-0.5">{t("experiment.vs_random")}</div>
          </div>
        )}
        <div className="rounded-xl border border-border/40 bg-background p-4 text-center">
          <div className="text-xs text-muted-foreground uppercase font-bold mb-1">{t("experiment_detail.ncd_best_acc")}</div>
          <div className="text-2xl font-black text-green-500">
            {bestClustering?.acc ? `${(bestClustering.acc * 100).toFixed(1)}%` : "—"}
          </div>
          <div className="text-[10px] text-muted-foreground mt-0.5">{bestClustering?.k ? `k=${bestClustering.k}` : ""}</div>
        </div>
        <div className="rounded-xl border border-border/40 bg-background p-4 text-center">
          <div className="text-xs text-muted-foreground uppercase font-bold mb-1">{t("experiment_detail.ncd_silhouette")}</div>
          <div className={`text-2xl font-black ${(silhouette ?? 0) > 0.1 ? "text-yellow-500" : "text-muted-foreground"}`}>
            {silhouette !== null ? silhouette.toFixed(3) : "—"}
          </div>
          <div className="text-[10px] text-muted-foreground mt-0.5">
            {clustering.kmeans_best?.k ? `k=${clustering.kmeans_best.k}` : ""}
          </div>
        </div>
      </div>

      {/* Ablation chart */}
      {ablation && <AblationChart ablation={ablation} />}

      {/* t-SNE viewer */}
      {scatterPoints.length > 0 && (
        <section>
          <h3 className="text-lg font-bold mb-4">
            {t("experiment.tsne")}
            <div className="inline-flex ml-3 gap-1">
              <button onClick={() => setTsneMode("2d")}
                className={`px-2 py-0.5 text-xs rounded font-bold ${tsneMode === "2d" ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
                {t("experiment_detail.ncd_tsne_toggle_2d")}
              </button>
              <button onClick={() => setTsneMode("3d")}
                className={`px-2 py-0.5 text-xs rounded font-bold ${tsneMode === "3d" ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
                {t("experiment_detail.ncd_tsne_toggle_3d")}
              </button>
            </div>
          </h3>
          <div className="rounded-xl border border-border/50 bg-background/50 p-4">
            {tsneMode === "2d" ? (
              <Scatter2D points={scatterPoints} speciesColors={speciesColors} />
            ) : tsne3d ? (
              <Scatter3D
                points={scatterPoints.map((p, i) => ({ ...p, z: tsne3d[i]?.[2] ?? 0 }))}
                species={[...new Set(scatterPoints.map((p) => p.species))]}
                speciesColors={speciesColors}
              />
            ) : (
              <p className="text-sm text-muted-foreground">{t("experiment.tsne_unavailable")}</p>
            )}
          </div>
        </section>
      )}

      {/* Clustering comparison chart */}
      {Object.keys(clustering).length > 0 && <ClusteringComparisonChart clustering={clustering} />}

      {/* Clustering table */}
      <section>
        <h3 className="text-lg font-bold mb-4">{t("experiment_detail.ncd_clustering")}</h3>
        <div className="overflow-x-auto rounded-xl border border-border/50">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50 text-xs uppercase text-muted-foreground font-bold">
                <th className="text-left py-3 px-4">{t("experiment_detail.ncd_algorithm")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_acc")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_nmi")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_ari")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_ami")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_f")}</th>
                <th className="text-center py-3 px-2">{t("experiment.k_label")}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30">
              {Object.entries(clustering).map(([key, val]) => (
                <tr key={key}>
                  <td className="py-2 px-4 font-bold">{key}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.acc?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.nmi?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.ari?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.ami?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.f_measure?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.k ?? val.n_clusters ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* OSR section */}
      <section>
        <h3 className="text-lg font-bold mb-4">{t("experiment_detail.ncd_osr")}</h3>

        {/* AUROC per class chart */}
        <PerClassAurocChart perClass={perClass} napoliAuroc={napoliAuroc} />

        {/* Per-class table */}
        {perClass.length > 0 && (
          <div className="overflow-x-auto rounded-xl border border-border/50 mt-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/50 text-xs uppercase text-muted-foreground font-bold">
                  <th className="text-left py-3 px-4">{t("experiment_detail.ncd_class")}</th>
                  <th className="text-center py-3 px-2">AUROC</th>
                  <th className="text-center py-3 px-2">{t("experiment_detail.ncd_known_acc")}</th>
                  <th className="text-center py-3 px-2">{t("experiment_detail.ncd_unknown_recall")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {perClass.map((pc) => (
                  <tr key={pc.left_out}
                    className={pc.left_out.toLowerCase().includes("majellensis") ? "bg-red-500/10 font-bold" : ""}>
                    <td className="py-2 px-4 font-bold">{pc.left_out}</td>
                    <td className="text-center py-2 px-2 font-mono">{(pc.auroc * 100).toFixed(1)}%</td>
                    <td className="text-center py-2 px-2 font-mono">{(pc.known_acc * 100).toFixed(1)}%</td>
                    <td className="text-center py-2 px-2 font-mono">{(pc.unknown_recall * 100).toFixed(1)}%</td>
                  </tr>
                ))}
                {napoliAuroc !== undefined && (
                  <tr className="bg-red-500/10 font-bold">
                    <td className="py-2 px-4 font-bold">
                      O. sphegodes_Napoli
                      <span className="ml-1.5 inline-block bg-red-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded">
                        {t("experiment_detail.ncd_novel_label")}
                      </span>
                    </td>
                    <td className="text-center py-2 px-2 font-mono text-red-500">{(napoliAuroc * 100).toFixed(1)}%</td>
                    <td className="text-center py-2 px-2 font-mono">—</td>
                    <td className="text-center py-2 px-2 font-mono">
                      {openset?.novel_class?.unknown_recall ? `${(openset.novel_class.unknown_recall * 100).toFixed(1)}%` : "—"}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

/* ---- Logs Panel ---- */
function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <Card>
      <CardContent className="pt-4 pb-3 text-center">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-xl font-bold text-primary">{(value * 100).toFixed(2)}%</p>
      </CardContent>
    </Card>
  );
}

function LogsPanel({ experimentId }: { experimentId: string }) {
  const { t } = useTranslation();
  const { data, isLoading, error } = useQuery({
    queryKey: ["log", experimentId],
    queryFn: () => fetchExperimentLog(experimentId),
    staleTime: 60 * 1000,
  });

  if (isLoading) return <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>;
  if (error) return <p className="text-sm text-muted-foreground text-center py-8">{t("experiment_detail.log_unavailable")}</p>;
  if (!data?.available) return (
    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
      <p className="text-sm">{t("experiment_detail.log_unavailable")}</p>
    </div>
  );

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-2">
        <p className="text-xs text-muted-foreground">{t("experiment_detail.log_size")} {(data.size / 1024).toFixed(1)} KB</p>
        <button
          onClick={() => navigator.clipboard.writeText(data.log ?? "")}
          className="text-xs px-2 py-1 rounded bg-muted hover:bg-muted/80"
        >
          {t("experiment_detail.log_copy")}
        </button>
      </div>
      <pre className="text-xs leading-relaxed bg-black/5 dark:bg-white/5 p-4 rounded-xl overflow-auto max-h-[600px] font-mono">
        {data.log}
      </pre>
    </div>
  );
}
