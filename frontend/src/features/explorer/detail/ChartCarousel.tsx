// frontend/src/features/explorer/detail/ChartCarousel.tsx

import * as React from "react";
import { useState, useCallback } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LossCurvesChart } from "@/features/explorer/components/LossCurvesChart";
import { MetricCurvesChart } from "@/features/explorer/components/MetricCurvesChart";
import { FoldSummaryChart } from "@/features/explorer/components/FoldSummaryChart";
import { ConfusionMatrixChart } from "@/features/explorer/components/ConfusionMatrixChart";
import { OverfittingProxyChart } from "@/features/explorer/components/OverfittingProxyChart";
import type { FoldData } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";


export interface ChartCarouselProps {
  folds: FoldData[];
  selectedFold: number | null;
  meanF1?: number | null;
  stdF1?: number | null;
  confusionMatrix: { matrix: number[][]; labels: string[] } | null;
  experimentId?: string;
}

export function ChartCarousel({ folds, selectedFold, meanF1, stdF1, confusionMatrix, experimentId }: ChartCarouselProps) {
  const { t } = useTranslation();
  const hasPhases = folds.length > 0 && folds[0]?.fold_phases?.length ? folds[0].fold_phases.length > 0 : false;
  const hasOverfitting = folds.some(f => f.history?.val_loss?.some((v: number) => v > 0));
  const [active, setActive] = useState(0);

  /* ---------- Phase 1 (SupCon) helper ---------- */
  const COLORS = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6", "#EF4444"];
  const supconData = hasPhases ? folds[0]?.fold_phases?.[0] : null;
  const supconEpochs = supconData?.epochs ?? 0;
  const ceEpochs = folds.length > 0 ? folds[0].history.train_loss.length : 0;
  const visibleFoldsSupcon = selectedFold === null ? folds : folds.filter((f) => f.fold === selectedFold);

  const supconChartData = React.useMemo(() => {
    if (!supconData) return [];
    return Array.from({ length: supconEpochs }, (_, ep) => {
      const row: Record<string, number | string> = { epoch: ep + 1 };
      for (const f of visibleFoldsSupcon) {
        const m = f.fold_phases?.[0]?.metrics ?? {};
        const tl = m["train_loss"]?.[ep];
        const ka = m["val_knn_acc"]?.[ep];
        if (tl !== undefined) row[`tl_${f.fold}`] = tl;
        if (ka !== undefined) row[`ka_${f.fold}`] = ka;
      }
      return row;
    });
  }, [visibleFoldsSupcon, supconData, supconEpochs]);

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
                  <CardTitle className="text-sm">{t("experiment_detail.supcon_loss_title")}</CardTitle>
                  <p className="text-[11px] text-muted-foreground mt-0.5">{t("experiment_detail.supcon_phases", { p1: supconEpochs, p2: ceEpochs, total: supconEpochs + ceEpochs })}</p>
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
                  {visibleFoldsSupcon.map((f, i) => (
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
                  <CardTitle className="text-sm">{t("experiment_detail.supcon_knn_title")}</CardTitle>
                  <p className="text-[11px] text-muted-foreground mt-0.5">{t("experiment_detail.supcon_phases", { p1: supconEpochs, p2: ceEpochs, total: supconEpochs + ceEpochs })}</p>
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
                  {visibleFoldsSupcon.map((f, i) => (
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
      case 2: return <FoldSummaryChart folds={folds} meanF1={meanF1} stdF1={stdF1} selectedFold={selectedFold} />;
      case 3: return hasOverfitting ? <OverfittingProxyChart folds={folds} selectedFold={selectedFold} /> : null;
      case 4: return confusionMatrix ? (
        <div className="space-y-2">
          {selectedFold !== null && (
            <p className="text-xs text-muted-foreground">
              {t("experiment_detail.charts.confusion_aggregate_note")}
            </p>
          )}
          <ConfusionMatrixChart matrix={confusionMatrix.matrix} labels={confusionMatrix.labels} />
        </div>
      ) : null;
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
