// frontend/src/features/explorer/components/OverfittingProxyChart.tsx

import { useMemo, useRef } from "react";
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
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { FoldData } from "@/lib/types";
import { foldToEpochPoints } from "../types";
import { useChartSize } from "../hooks/useChartSize";
import { useChartColors } from "../hooks/useChartColors";

interface OverfittingProxyChartProps {
  folds: FoldData[];
  selectedFold: number | null;
}

export function OverfittingProxyChart({ folds, selectedFold }: OverfittingProxyChartProps) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(ref);
  const chartColors = useChartColors();

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
        <CardTitle className="text-base">{t("experiment_detail.charts.overfitting_title")}</CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <div ref={ref} style={{ width: "100%", height }}>
          <LineChart data={merged} width={width} height={height}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis dataKey="epoch" tick={{ fontSize: 14 }} />
            <YAxis tick={{ fontSize: 14 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 13 }} align="right" />
            <ReferenceLine
              y={1}
              stroke={chartColors[4]}
              strokeDasharray="4 4"
              label={{ value: t("experiment_detail.charts.overfitting_threshold"), fontSize: 13, fill: chartColors[4] }}
            />
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
            {visibleFolds.map((f) => (
              <Line
                key={`ratio_${f.fold}`}
                dataKey={`ratio_${f.fold}`}
                name={`${t("experiment_detail.fold_selector.fold")}${f.fold}`}
                stroke={chartColors[f.fold % chartColors.length]}
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          {t("experiment_detail.charts.overfitting_description")}
        </p>
      </CardContent>
    </Card>
  );
}
