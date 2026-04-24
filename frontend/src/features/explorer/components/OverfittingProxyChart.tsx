// frontend/src/features/explorer/components/OverfittingProxyChart.tsx

import { useMemo, useRef } from "react";
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
import { foldToEpochPoints, FOLD_COLORS } from "../types";
import { useChartSize } from "../hooks/useChartSize";

interface OverfittingProxyChartProps {
  folds: FoldData[];
  selectedFold: number | null;
}

export function OverfittingProxyChart({ folds, selectedFold }: OverfittingProxyChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(ref);

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
        if (i < points.length && points[i].overfitting_ratio !== undefined) {
          row[`ratio_${f.fold}`] = points[i].overfitting_ratio!;
        }
      });
      return row;
    });
  }, [visibleFolds]);

  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-base">Overfitting Proxy (val_loss / train_loss)</CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <div ref={ref} style={{ width: "100%", height }}>
          <LineChart data={merged} width={width} height={height}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
            <XAxis dataKey="epoch" tick={{ fontSize: 14 }} />
            <YAxis tick={{ fontSize: 14 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 13 }} align="right" />
            <ReferenceLine
              y={1}
              stroke="#EF4444"
              strokeDasharray="4 4"
              label={{ value: "Overfitting threshold", fontSize: 13, fill: "#EF4444" }}
            />
            {visibleFolds.map((f) => (
              <Line
                key={`ratio_${f.fold}`}
                dataKey={`ratio_${f.fold}`}
                name={`Fold ${f.fold}`}
                stroke={FOLD_COLORS[(f.fold - 1) % FOLD_COLORS.length]}
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Ratio of val_loss / train_loss per epoch. Values above 1 indicate potential overfitting.
        </p>
      </CardContent>
    </Card>
  );
}
