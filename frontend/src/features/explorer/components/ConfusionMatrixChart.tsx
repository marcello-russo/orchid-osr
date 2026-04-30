// frontend/src/features/explorer/components/ConfusionMatrixChart.tsx

import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useChartSize } from "@/features/explorer/hooks/useChartSize";

interface ConfusionMatrixChartProps {
  matrix: number[][];
  labels: string[];
}

export function ConfusionMatrixChart({ matrix, labels }: ConfusionMatrixChartProps) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const { width } = useChartSize(containerRef);

  if (!matrix.length) return null;

  const maxVal = Math.max(...matrix.flat());
  const numCols = labels.length;

  // Dynamic sizing based on container width
  const labelW = width < 500 ? 100 : width < 700 ? 140 : 200;
  const availableWidth = Math.max(width - labelW - 16, 200);
  const cellW = Math.max(Math.floor(availableWidth / numCols), 40);
  const cellH = width < 500 ? 50 : width < 700 ? 60 : 80;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{t("experiment_detail.charts.confusion_title")}</CardTitle>
      </CardHeader>
      <CardContent>
        <TooltipProvider>
          <div ref={containerRef} className="overflow-x-auto">
            {/* Column headers */}
            <div className="flex" style={{ paddingLeft: labelW + 8 }}>
              {labels.map((label) => (
                <div
                  key={label}
                  className="text-xs sm:text-sm text-muted-foreground text-center overflow-hidden"
                  style={{ width: cellW, minWidth: cellW }}
                >
                  {label.replace("O. ", "")}
                </div>
              ))}
            </div>

            {/* Rows */}
            {matrix.map((row, i) => (
              <div key={i} className="flex items-center">
                {/* Row label */}
                <div
                  className="text-xs sm:text-sm text-muted-foreground text-right pr-2 overflow-hidden"
                  style={{ width: labelW, minWidth: labelW }}
                >
                  {labels[i].replace("O. ", "")}
                </div>

                {/* Cells */}
                {row.map((val, j) => {
                  const intensity = maxVal > 0 ? val / maxVal : 0;
                  const isCorrect = i === j;

                  return (
                    <Tooltip key={j}>
                      <TooltipTrigger>
                        <div
                          style={{ width: cellW, minWidth: cellW, height: cellH }}
                          className={cn(
                            "flex items-center justify-center text-xs sm:text-sm font-mono border border-border transition-colors duration-150 cursor-default",
                            isCorrect && intensity > 0.7
                              ? "bg-primary/30 text-primary font-bold"
                              : intensity > 0.1
                                ? "bg-primary/10"
                                : "bg-transparent",
                            !isCorrect && val > 0 && "bg-accent/20 text-accent"
                          )}
                        >
                          {val}
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="text-xs">
                          {t("experiment_detail.charts.confusion_true")} {labels[i]} → {t("experiment_detail.charts.confusion_pred")} {labels[j]}: {val}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  );
                })}
              </div>
            ))}
          </div>
        </TooltipProvider>
      </CardContent>
    </Card>
  );
}
