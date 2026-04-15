// frontend/src/features/explorer/components/FoldSummaryChart.tsx

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { FoldData } from "@/lib/types";

interface FoldSummaryChartProps {
  folds: FoldData[];
  meanF1?: number | null;
  stdF1?: number | null;
}

export function FoldSummaryChart({ folds, meanF1, stdF1 }: FoldSummaryChartProps) {
  const data = folds.map((f) => ({
    name: `Fold ${f.fold}`,
    F1: f.best_f1,
    Accuracy: f.best_accuracy,
  }));

  const titleSuffix =
    meanF1 != null
      ? ` — Mean F1: ${(meanF1 * 100).toFixed(1)}%${
          stdF1 != null ? ` ± ${(stdF1 * 100).toFixed(1)}%` : ""
        }`
      : "";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Fold Summary{titleSuffix}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 12 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={60} />
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
            <Bar dataKey="F1" fill="#3B82F6" radius={[0, 4, 4, 0]} barSize={16} />
            <Bar dataKey="Accuracy" fill="#D97706" radius={[0, 4, 4, 0]} barSize={16} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
