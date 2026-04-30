// frontend/src/features/explorer/components/LossCurvesChart.tsx

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

interface LossCurvesChartProps {
  folds: FoldData[];
  selectedFold: number | null;
}

export function LossCurvesChart({ folds, selectedFold }: LossCurvesChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(ref);
  const chartColors = useChartColors();
  const [mode, setMode] = useState<"train" | "val">("val");

  const visibleFolds = useMemo(
    () => selectedFold === null ? folds : folds.filter((f) => f.fold === selectedFold),
    [folds, selectedFold]
  );

  const merged = useMemo(() => {
    const allEpochs = visibleFolds.map((f) => ({
      fold: f,
      points: foldToEpochPoints(f),
    }));
    const maxLen = Math.max(...allEpochs.map((e) => e.points.length), 0);
    return Array.from({ length: maxLen }, (_, i) => {
      const row: Record<string, number> = { epoch: i + 1 };
      allEpochs.forEach(({ fold: f, points }) => {
        if (i < points.length) {
          row[`${f.fold}`] = mode === "train" ? points[i].train_loss! : points[i].val_loss!;
        }
      });
      return row;
    });
  }, [visibleFolds, mode]);

  const maxLen = merged.length;

  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Loss Curves</CardTitle>
          <div className="flex gap-1">
            {(["train", "val"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "rounded-md px-2 py-0.5 text-xs font-medium transition-colors cursor-pointer",
                  mode === m
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground bg-muted/50 hover:bg-muted"
                )}
              >
                {m === "train" ? "Train" : "Validation"}
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
            <YAxis tick={{ fontSize: 14 }} />
            <Tooltip
              formatter={(value) => Number(value).toFixed(2)}
              labelFormatter={(l) => `Epoch ${l}`}
            />
            <Legend
              wrapperStyle={{ fontSize: 13 }}
              formatter={(value: string) => `Fold ${value}`}
              align="right"
            />
            {visibleFolds.map((f) => {
              const color = chartColors[f.fold % chartColors.length];
              return (
                <Line
                  key={`loss_${f.fold}`}
                  dataKey={`${f.fold}`}
                  name={`${f.fold}`}
                  stroke={color}
                  dot={false}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                />
              );
            })}
            {visibleFolds.map((f) => {
              if (f.best_epoch <= 0 || f.best_epoch > maxLen) return null;
              const color = chartColors[f.fold % chartColors.length];
              return (
                <ReferenceLine
                  key={`best_${f.fold}`}
                  x={f.best_epoch}
                  stroke={color}
                  strokeDasharray="2 2"
                  strokeOpacity={0.4}
                />
              );
            })}
          </LineChart>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Training or validation loss per epoch for each fold. Dashed vertical lines mark the best epoch.
        </p>
      </CardContent>
    </Card>
  );
}
