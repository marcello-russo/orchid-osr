// frontend/src/features/explorer/components/OverfittingProxyChart.tsx

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { FoldData } from "@/lib/types";
import { foldToEpochPoints, FOLD_COLORS } from "../types";

interface OverfittingProxyChartProps {
  folds: FoldData[];
  selectedFold: number | null;
}

export function OverfittingProxyChart({ folds, selectedFold }: OverfittingProxyChartProps) {
  const visibleFolds = selectedFold === null
    ? folds
    : folds.filter((f) => f.fold === selectedFold);

  const allEpochs = visibleFolds.map((f) => ({
    fold: f,
    points: foldToEpochPoints(f),
  }));
  const maxLen = Math.max(...allEpochs.map((e) => e.points.length), 0);

  const merged = Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number> = { epoch: i + 1 };
    allEpochs.forEach(({ fold: f, points }) => {
      if (i < points.length && points[i].overfitting_ratio !== undefined) {
        row[`ratio_${f.fold}`] = points[i].overfitting_ratio!;
      }
    });
    return row;
  });

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Overfitting Proxy (val_loss / train_loss)</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={merged}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: 12,
              }}
            />
            <Legend />
            <ReferenceLine
              y={1}
              stroke="#EF4444"
              strokeDasharray="4 4"
              label={{ value: "Overfitting threshold", fontSize: 10, fill: "#EF4444" }}
            />
            {visibleFolds.map((f, i) => (
              <Line
                key={`ratio_${f.fold}`}
                dataKey={`ratio_${f.fold}`}
                name={`Fold ${f.fold}`}
                stroke={FOLD_COLORS[i % FOLD_COLORS.length]}
                dot={false}
                strokeWidth={1.5}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
