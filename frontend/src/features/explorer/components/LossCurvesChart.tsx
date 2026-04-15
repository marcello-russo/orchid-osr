// frontend/src/features/explorer/components/LossCurvesChart.tsx

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

interface LossCurvesChartProps {
  folds: FoldData[];
  selectedFold: number | null;
}

export function LossCurvesChart({ folds, selectedFold }: LossCurvesChartProps) {
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
      if (i < points.length) {
        row[`train_${f.fold}`] = points[i].train_loss!;
        row[`val_${f.fold}`] = points[i].val_loss!;
      }
    });
    return row;
  });

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Loss Curves</CardTitle>
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
            {visibleFolds.map((f, i) => {
              const color = FOLD_COLORS[i % FOLD_COLORS.length];
              return [
                <Line
                  key={`train_${f.fold}`}
                  dataKey={`train_${f.fold}`}
                  name={`Fold ${f.fold} Train`}
                  stroke={color}
                  dot={false}
                  strokeWidth={1.5}
                />,
                <Line
                  key={`val_${f.fold}`}
                  dataKey={`val_${f.fold}`}
                  name={`Fold ${f.fold} Val`}
                  stroke={color}
                  dot={false}
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                />,
                f.best_epoch > 0 && f.best_epoch <= maxLen && (
                  <ReferenceLine
                    key={`best_${f.fold}`}
                    x={f.best_epoch}
                    stroke={color}
                    strokeDasharray="2 2"
                    strokeOpacity={0.4}
                  />
                ),
              ];
            })}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
