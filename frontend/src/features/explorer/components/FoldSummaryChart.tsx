// frontend/src/features/explorer/components/FoldSummaryChart.tsx

import { useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { FoldData } from "@/lib/types";
import { useChartSize } from "../hooks/useChartSize";
import { useChartColors } from "../hooks/useChartColors";

interface FoldSummaryChartProps {
  folds: FoldData[];
  meanF1?: number | null;
  stdF1?: number | null;
}

export function FoldSummaryChart({ folds, meanF1, stdF1 }: FoldSummaryChartProps) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(ref);
  const chartColors = useChartColors();

  const data = useMemo(
    () => folds.map((f) => ({
      name: `${t("experiment_detail.fold_selector.fold")}${f.fold}`,
      F1: f.best_f1,
      Accuracy: f.best_accuracy,
    })),
    [folds, t]
  );

  const titleSuffix =
    meanF1 != null
      ? ` — ${t("experiment_detail.charts.fold_mean_f1")}${(meanF1 * 100).toFixed(2)}%${
          stdF1 != null ? ` ± ${(stdF1 * 100).toFixed(2)}%` : ""
        }`
      : "";

  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-base">{t("experiment_detail.charts.folds")}{titleSuffix}</CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <div ref={ref} style={{ width: "100%", height }}>
          <BarChart data={data} layout="vertical" width={width} height={height}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 14 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 14 }} width={80} />
            <Tooltip
              formatter={(value) => (Number(value) * 100).toFixed(2) + "%"}
            />
            <Legend wrapperStyle={{ fontSize: 13 }} align="right" />
            <Bar dataKey="F1" fill={chartColors[0]} radius={[0, 4, 4, 0]} barSize={20} isAnimationActive={false} />
            <Bar dataKey="Accuracy" fill={chartColors[1]} radius={[0, 4, 4, 0]} barSize={20} isAnimationActive={false} />
          </BarChart>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          {t("experiment_detail.charts.fold_summary_description")}
        </p>
      </CardContent>
    </Card>
  );
}
