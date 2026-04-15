// frontend/src/features/explorer/components/ConfusionMatrixChart.tsx

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ConfusionMatrixChartProps {
  matrix: number[][];
  labels: string[];
}

export function ConfusionMatrixChart({ matrix, labels }: ConfusionMatrixChartProps) {
  if (!matrix.length) return null;

  const maxVal = Math.max(...matrix.flat());
  const n = matrix.length;
  const cellSize = Math.min(60, Math.floor(480 / n));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Confusion Matrix (Test Set)</CardTitle>
      </CardHeader>
      <CardContent>
        <TooltipProvider>
          <div className="overflow-x-auto">
            {/* Column headers */}
            <div className="flex" style={{ paddingLeft: cellSize + 8 }}>
              {labels.map((label) => (
                <div
                  key={label}
                  className="text-xs text-muted-foreground text-center truncate"
                  style={{ width: cellSize, minWidth: cellSize }}
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
                  className="text-xs text-muted-foreground text-right pr-2 truncate"
                  style={{ width: cellSize, minWidth: cellSize }}
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
                          style={{ width: cellSize, minWidth: cellSize, height: cellSize }}
                          className={cn(
                            "flex items-center justify-center text-xs font-mono border border-border transition-colors duration-150 cursor-default",
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
                          True: {labels[i]} → Pred: {labels[j]}: {val}
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
