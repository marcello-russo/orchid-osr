// frontend/src/features/explorer/detail/BaselineResults.tsx

import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
} from "recharts";
import { useRef } from "react";
import { useChartSize } from "@/features/explorer/hooks/useChartSize";
import { useChartColors } from "@/features/explorer/hooks/useChartColors";

export interface TestMetrics {
  accuracy: number;
  f1_macro: number;
  precision_macro?: number;
  recall_macro?: number;
  per_class_metrics?: Record<string, { precision: number; recall: number; f1: number; support: number }>;
}

export function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <Card>
      <CardContent className="pt-4 pb-3 text-center">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-xl font-bold text-primary">{(value * 100).toFixed(2)}%</p>
      </CardContent>
    </Card>
  );
}

export function BaselineResults({ metrics }: { metrics: TestMetrics }) {
  const { t } = useTranslation();
  const chartRef = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(chartRef);
  const chartColors = useChartColors();

  const classData = Object.entries(metrics.per_class_metrics ?? {}).map(([className, c]) => ({
    name: className.replace("O. ", ""),
    F1: c.f1,
    Precision: c.precision,
    Recall: c.recall,
  }));

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label={t("experiment_detail.metrics.accuracy")} value={metrics.accuracy} />
        <MetricCard label={t("experiment_detail.metrics.f1")} value={metrics.f1_macro} />
        {metrics.precision_macro != null && <MetricCard label={t("experiment_detail.metrics.precision")} value={metrics.precision_macro} />}
        {metrics.recall_macro != null && <MetricCard label={t("experiment_detail.metrics.recall")} value={metrics.recall_macro} />}
      </div>

      {/* Per-class bar chart */}
      {classData.length > 0 && (
        <Card>
          <CardHeader className="pb-2 border-b border-border/50">
            <CardTitle className="text-base">{t("experiment_detail.per_class_metrics")}</CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            <div ref={chartRef} style={{ width: "100%", height }}>
              <BarChart data={classData} width={width} height={height}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
                <XAxis dataKey="name" tick={{ fontSize: 13 }} />
                <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 14 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(value) => (Number(value) * 100).toFixed(2) + "%"} />
                <Bar dataKey="F1" fill={chartColors[0]} radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
                <Bar dataKey="Precision" fill={chartColors[1]} radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
                <Bar dataKey="Recall" fill={chartColors[2]} radius={[4, 4, 0, 0]} barSize={24} isAnimationActive={false} />
              </BarChart>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Per-class table */}
      {classData.length > 0 && (
        <Card>
          <CardHeader className="pb-2 border-b border-border/50">
            <CardTitle className="text-base">{t("experiment_detail.per_class_breakdown")}</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("experiment_detail.table.class")}</TableHead>
                  <TableHead className="text-right">{t("experiment_detail.metrics.precision")}</TableHead>
                  <TableHead className="text-right">{t("experiment_detail.metrics.recall")}</TableHead>
                  <TableHead className="text-right">{t("experiment_detail.charts.f1_label")}</TableHead>
                  <TableHead className="text-right whitespace-nowrap">{t("experiment_detail.table.support")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(metrics.per_class_metrics!).map(([className, c]) => (
                  <TableRow key={className}>
                    <TableCell className="font-medium">{className}</TableCell>
                    <TableCell className="text-right font-mono">{(c.precision * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right font-mono">{(c.recall * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right font-mono">{(c.f1 * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right whitespace-nowrap">{c.support}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
