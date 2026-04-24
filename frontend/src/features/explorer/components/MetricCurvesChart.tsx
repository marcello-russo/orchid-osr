// frontend/src/features/explorer/components/MetricCurvesChart.tsx

import { useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
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
          <CardTitle className="text-base">Metric Curves</CardTitle>
          <div className="flex gap-1">
            {(["f1", "accuracy"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMetric(m)}
                className={cn(
                  "rounded-md px-2 py-0.5 text-xs font-medium transition-colors",
                  metric === m
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground bg-muted/50 hover:bg-muted"
                )}
              >
                {m === "f1" ? "F1" : "Accuracy"}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <div ref={ref} style={{ width: "100%", height }}>
          <LineChart data={merged} width={width} height={height}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis dataKey="epoch" tick={{ fontSize: 14 }} />
            <YAxis tick={{ fontSize: 14 }} domain={[0, 1]} />
            <Tooltip
              formatter={(value) => (Number(value) * 100).toFixed(2) + "%"}
              labelFormatter={(l) => `Epoch ${l}`}
            />
            <Legend
              wrapperStyle={{ fontSize: 13 }}
              formatter={(value: string) => `Fold ${value}`}
              align="right"
            />
            {metric === "f1" && meanBestF1 > 0 && (
              <ReferenceLine
                y={meanBestF1}
                stroke={chartColors[1]}
                strokeDasharray="4 4"
                label={{ value: `Mean Best F1: ${(meanBestF1 * 100).toFixed(2)}%`, fontSize: 13, fill: chartColors[1] }}
              />
            )}
            {visibleFolds.map((f) => (
              <Line
                key={`metric_${f.fold}`}
                dataKey={`${f.fold}`}
                name={`${f.fold}`}
                stroke={chartColors[(f.fold - 1) % chartColors.length]}
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Selected metric per epoch for each fold. The dashed horizontal line shows the mean best F1.
        </p>
      </CardContent>
    </Card>
  );
}
