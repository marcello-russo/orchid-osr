// frontend/src/features/explorer/components/MetricCurvesChart.tsx

import { useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { FoldData } from "@/lib/types";
import { foldToEpochPoints } from "../types";
import { useChartSize } from "../hooks/useChartSize";
import { useChartColors } from "../hooks/useChartColors";
import { cn } from "@/lib/utils";

interface MetricCurvesChartProps {
  folds: FoldData[];
  selectedFold: number | null;
}

export function MetricCurvesChart({ folds, selectedFold }: MetricCurvesChartProps) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(ref);
  const chartColors = useChartColors();
  const [metric, setMetric] = useState<"f1" | "accuracy">("f1");

  const visibleFolds = useMemo(
    () => selectedFold === null ? folds : folds.filter((f) => f.fold === selectedFold),
    [folds, selectedFold]
  );

  const { merged, meanBestF1 } = useMemo(() => {
    const allEpochs = visibleFolds.map((f) => ({
      fold: f,
      points: foldToEpochPoints(f),
    }));
    const maxLen = Math.max(...allEpochs.map((e) => e.points.length), 0);
    const data = Array.from({ length: maxLen }, (_, i) => {
      const row: Record<string, number> = { epoch: i + 1 };
      allEpochs.forEach(({ fold: f, points }) => {
        if (i < points.length) {
          row[`${f.fold}`] = metric === "f1" ? points[i].val_f1! : points[i].val_accuracy!;
        }
      });
      return row;
    });
    const mean = visibleFolds.length > 0
      ? visibleFolds.reduce((s, f) => s + f.best_f1, 0) / visibleFolds.length
      : 0;
    return { merged: data, meanBestF1: mean };
  }, [visibleFolds, metric]);

  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{t("experiment_detail.charts.metrics")}</CardTitle>
          <div className="flex gap-1">
            {(["f1", "accuracy"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMetric(m)}
                className={cn(
                  "rounded-md px-2 py-0.5 text-xs font-medium transition-colors cursor-pointer",
                  metric === m
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground bg-muted/50 hover:bg-muted"
                )}
              >
                {m === "f1" ? t("experiment_detail.charts.f1_label") : t("experiment_detail.charts.accuracy")}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <div ref={ref} style={{ width: "100%", height: height || 400, minHeight: 350 }}>
          <ResponsiveContainer width="100%" height={height || 400}>
          <LineChart data={merged}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis dataKey="epoch" tick={{ fontSize: 14 }} />
            <YAxis tick={{ fontSize: 14 }} domain={[0, 1]} />
            <Tooltip
              formatter={(value) => (Number(value) * 100).toFixed(2) + "%"}
              labelFormatter={(l) => `${t("experiment_detail.charts.epoch")}${l}`}
            />
            <Legend
              wrapperStyle={{ fontSize: 13 }}
              formatter={(value: string) => `${t("experiment_detail.fold_selector.fold")}${value}`}
              align="right"
            />
            {metric === "f1" && meanBestF1 > 0 && (
              <ReferenceLine
                y={meanBestF1}
                stroke={chartColors[1]}
                strokeDasharray="4 4"
                label={{ value: `${t("experiment_detail.charts.mean_best_f1")}${(meanBestF1 * 100).toFixed(2)}%`, fontSize: 13, fill: chartColors[1] }}
              />
            )}
            {visibleFolds.map((f) => (
              <Line
                key={`metric_${f.fold}`}
                dataKey={`${f.fold}`}
                name={`${f.fold}`}
                stroke={chartColors[f.fold % chartColors.length]}
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            ))}
            {visibleFolds.length > 0 && visibleFolds[0].phase_epochs?.map((ep, i) => {
              if (i === 0 || ep <= 0 || ep > (merged.length || 1)) return null;
              return (
                <ReferenceLine
                  key={`phase_${i}`}
                  x={ep}
                  stroke="var(--muted-foreground)"
                  strokeDasharray="4 4"
                  strokeOpacity={0.6}
                  label={visibleFolds[0].phase_names?.[i] ?? `Phase ${i+1}`}
                />
              );
            })}
          </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          {t("experiment_detail.charts.metrics_description")}
        </p>
      </CardContent>
    </Card>
  );
}
