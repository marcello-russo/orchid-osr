// frontend/src/pages/ExperimentDetailPage.tsx

import { useState, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useExperiment } from "@/features/explorer/hooks/useExperiment";
import { FoldSelector } from "@/features/explorer/components/FoldSelector";
import { LossCurvesChart } from "@/features/explorer/components/LossCurvesChart";
import { MetricCurvesChart } from "@/features/explorer/components/MetricCurvesChart";
import { FoldSummaryChart } from "@/features/explorer/components/FoldSummaryChart";
import { ConfusionMatrixChart } from "@/features/explorer/components/ConfusionMatrixChart";
import { OverfittingProxyChart } from "@/features/explorer/components/OverfittingProxyChart";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { useRef } from "react";
import { useChartSize } from "@/features/explorer/hooks/useChartSize";

export function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data: experiment, isLoading, error } = useExperiment(id ?? "");
  const [selectedFold, setSelectedFold] = useState<number | null>(null);

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
        <p>Experiment not found.</p>
        <Link to="/experiments" className="mt-2 inline-block text-primary hover:underline">
          Back to experiments
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
          Experiments
        </Link>
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold text-primary">{id}</h1>
          <Badge variant="secondary" className="text-xs">
            {summary.type}
          </Badge>
          {summary.model_variant && (
            <Badge variant="outline" className="font-mono text-xs">
              {summary.model_variant}
            </Badge>
          )}
        </div>

        {/* Config summary */}
        <div className="mt-2 flex flex-wrap gap-4 text-sm text-muted-foreground">
          {config.num_epochs && <span>Epochs: {config.num_epochs}</span>}
          {config.batch_size && <span>Batch: {config.batch_size}</span>}
          {config.learning_rate && <span>LR: {config.learning_rate}</span>}
          {config.patience !== undefined && <span>Patience: {config.patience}</span>}
          {summary.test_accuracy !== null && (
            <span className="font-medium text-primary">
              Test Accuracy: {(summary.test_accuracy * 100).toFixed(2)}%
            </span>
          )}
          {summary.test_f1 !== null && (
            <span className="font-medium text-primary">
              Test F1: {(summary.test_f1 * 100).toFixed(2)}%
            </span>
          )}
        </div>
      </div>

      {/* Fold selector — only for kfold experiments */}
      {hasFolds && (
        <FoldSelector
          folds={folds}
          selectedFold={selectedFold}
          onSelect={setSelectedFold}
        />
      )}

      {/* Charts carousel */}
      {hasFolds ? (
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
        />
      ) : hasResults && results.test_metrics ? (
        <BaselineResults metrics={results.test_metrics} />
      ) : (
        <div className="py-10 text-center text-muted-foreground">
          Baseline experiment — no results data available.
        </div>
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
}

const CHART_TITLES = [
  "Loss Curves",
  "Metric Curves",
  "Fold Summary",
  "Overfitting Proxy",
  "Confusion Matrix",
];

function ChartCarousel({ folds, selectedFold, meanF1, stdF1, confusionMatrix }: ChartCarouselProps) {
  const total = confusionMatrix ? 5 : 4;
  const [active, setActive] = useState(0);

  const clamp = useCallback((i: number) => Math.max(0, Math.min(total - 1, i)), [total]);

  return (
    <div className="space-y-3">
      {/* Nav bar — chip buttons */}
      <div className="flex items-center gap-2 overflow-x-auto scrollbar-none">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setActive((a) => clamp(a - 1))}
          disabled={active === 0}
          className="shrink-0 hidden sm:flex"
        >
          <ChevronLeft className="size-4" />
        </Button>
        {Array.from({ length: total }, (_, i) => (
          <button
            key={i}
            onClick={() => setActive(i)}
            className={cn(
              "shrink-0 rounded-full px-4 py-2 text-sm font-medium transition-colors whitespace-nowrap min-h-[44px]",
              i === active
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground bg-muted/50 hover:bg-muted"
            )}
          >
            {CHART_TITLES[i]}
          </button>
        ))}
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setActive((a) => clamp(a + 1))}
          disabled={active >= total - 1}
          className="shrink-0 hidden sm:flex"
        >
          <ChevronRight className="size-4" />
        </Button>
      </div>

      {/* Chart area — only active chart is rendered */}
      <div>
        {active === 0 && <LossCurvesChart folds={folds} selectedFold={selectedFold} />}
        {active === 1 && <MetricCurvesChart folds={folds} selectedFold={selectedFold} />}
        {active === 2 && <FoldSummaryChart folds={folds} meanF1={meanF1} stdF1={stdF1} />}
        {active === 3 && <OverfittingProxyChart folds={folds} selectedFold={selectedFold} />}
        {active === 4 && confusionMatrix && (
          <ConfusionMatrixChart matrix={confusionMatrix.matrix} labels={confusionMatrix.labels} />
        )}
      </div>
    </div>
  );
}

/* ---- Baseline Results ---- */

interface TestMetrics {
  accuracy: number;
  f1_macro: number;
  precision_macro?: number;
  recall_macro?: number;
  per_class?: { class: string; precision: number; recall: number; f1: number; support: number }[];
}

const FOLD_COLORS = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6", "#EF4444", "#EC4899"];

function BaselineResults({ metrics }: { metrics: TestMetrics }) {
  const chartRef = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(chartRef);

  const classData = (metrics.per_class ?? []).map((c, i) => ({
    name: c.class.replace("O. ", ""),
    F1: c.f1,
    Precision: c.precision,
    Recall: c.recall,
    fill: FOLD_COLORS[i % FOLD_COLORS.length],
  }));

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Accuracy" value={metrics.accuracy} />
        <MetricCard label="F1 Macro" value={metrics.f1_macro} />
        {metrics.precision_macro != null && <MetricCard label="Precision" value={metrics.precision_macro} />}
        {metrics.recall_macro != null && <MetricCard label="Recall" value={metrics.recall_macro} />}
      </div>

      {/* Per-class bar chart */}
      {classData.length > 0 && (
        <Card>
          <CardHeader className="pb-2 border-b border-border/50">
            <CardTitle className="text-base">Per-Class Metrics</CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            <div ref={chartRef} style={{ width: "100%", height }}>
              <BarChart data={classData} width={width} height={height}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
                <XAxis dataKey="name" tick={{ fontSize: 13 }} />
                <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 14 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(value) => (Number(value) * 100).toFixed(2) + "%"} />
                <Bar dataKey="F1" fill="#3B82F6" radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
                <Bar dataKey="Precision" fill="#D97706" radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
                <Bar dataKey="Recall" fill="#10B981" radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
              </BarChart>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Per-class table */}
      {classData.length > 0 && (
        <Card>
          <CardHeader className="pb-2 border-b border-border/50">
            <CardTitle className="text-base">Per-Class Breakdown</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="sticky left-0 bg-card z-10">Class</TableHead>
                  <TableHead className="text-right">Precision</TableHead>
                  <TableHead className="text-right">Recall</TableHead>
                  <TableHead className="text-right">F1</TableHead>
                  <TableHead className="text-right hidden sm:table-cell">Support</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {metrics.per_class!.map((c) => (
                  <TableRow key={c.class}>
                    <TableCell className="font-medium sticky left-0 bg-card z-10">{c.class}</TableCell>
                    <TableCell className="text-right font-mono">{(c.precision * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right font-mono">{(c.recall * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right font-mono">{(c.f1 * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right hidden sm:table-cell">{c.support}</TableCell>
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
