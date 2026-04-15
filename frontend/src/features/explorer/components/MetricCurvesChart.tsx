// frontend/src/features/explorer/components/MetricCurvesChart.tsx

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

interface MetricCurvesChartProps {
  folds: FoldData[];
  selectedFold: number | null;
}

export function MetricCurvesChart({ folds, selectedFold }: MetricCurvesChartProps) {
  const visibleFolds = selectedFold === null
    ? folds
    : folds.filter((f) => f.fold === selectedFold);

  const allEpochs = visibleFolds.map((f) => ({
    fold: f,
    points: foldToEpochPoints(f),
  }));
  const maxLen = Math.max(...allEpochs.map((e) => e.points.length), 0);

  const meanBestF1 =
    visibleFolds.length > 0
      ? visibleFolds.reduce((sum, f) => sum + f.best_f1, 0) / visibleFolds.length
      : 0;

  const merged = Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number> = { epoch: i + 1 };
    allEpochs.forEach(({ fold: f, points }) => {
      if (i < points.length) {
        row[`f1_${f.fold}`] = points[i].val_f1!;
        row[`acc_${f.fold}`] = points[i].val_accuracy!;
      }
    });
    return row;
  });

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Metric Curves</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={merged}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: 12,
              }}
              formatter={(value) => (Number(value) * 100).toFixed(1) + "%"}
            />
            <Legend />
            {meanBestF1 > 0 && (
              <ReferenceLine
                y={meanBestF1}
                stroke="#D97706"
                strokeDasharray="4 4"
                label={{ value: `Mean Best F1: ${(meanBestF1 * 100).toFixed(1)}%`, fontSize: 10, fill: "#D97706" }}
              />
            )}
            {visibleFolds.map((f, i) => {
              const color = FOLD_COLORS[i % FOLD_COLORS.length];
              return [
                <Line
                  key={`f1_${f.fold}`}
                  dataKey={`f1_${f.fold}`}
                  name={`Fold ${f.fold} F1`}
                  stroke={color}
                  dot={false}
                  strokeWidth={1.5}
                />,
                <Line
                  key={`acc_${f.fold}`}
                  dataKey={`acc_${f.fold}`}
                  name={`Fold ${f.fold} Accuracy`}
                  stroke={color}
                  dot={false}
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                />,
              ];
            })}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
