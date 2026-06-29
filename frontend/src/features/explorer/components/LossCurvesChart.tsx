// frontend/src/features/explorer/components/LossCurvesChart.tsx

import { useMemo, useRef, useState } from "react";
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
import { cn } from "@/lib/utils";

interface LossCurvesChartProps {
  folds: FoldData[];
  selectedFold: number | null;
}

export function LossCurvesChart({ folds, selectedFold }: LossCurvesChartProps) {
  const { t } = useTranslation();
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

  // Debug: mostra info dati in alto a destra
  const debugInfo = `${visibleFolds.length} folds, ${maxLen} epochs, mode=${mode}`;

  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{t("experiment_detail.charts.loss")}</CardTitle>
          <span className="text-[10px] text-muted-foreground font-mono">{debugInfo}</span>
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
                {m === "train" ? t("experiment_detail.charts.train") : t("experiment_detail.charts.validation")}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <div ref={ref} style={{ width: "100%", height: height || 400, minHeight: 350 }}>
          <LineChart data={merged} width={width || 800} height={height || 400}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis dataKey="epoch" tick={{ fontSize: 14 }} />
            <YAxis tick={{ fontSize: 14 }} domain={['auto', 'auto']} />
            <Tooltip
              formatter={(value) => Number(value).toFixed(2)}
              labelFormatter={(l) => `${t("experiment_detail.charts.epoch")}${l}`}
            />
            <Legend
              wrapperStyle={{ fontSize: 13 }}
              formatter={(value: string) => `${t("experiment_detail.fold_selector.fold")}${value}`}
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
            {visibleFolds.length > 0 && visibleFolds[0].phase_epochs?.map((ep, i) => {
              if (i === 0 || ep <= 0 || ep > maxLen) return null;
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
          </LineChart>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          {t("experiment_detail.charts.loss_description")}
        </p>
      </CardContent>
    </Card>
  );
}
