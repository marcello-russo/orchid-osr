import { useRef, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartSize } from "@/features/explorer/hooks/useChartSize";

interface ConfusionMatrixChartProps {
  matrix: number[][];
  labels: string[];
  nMax?: number;
}

const PURPLE = { r: 139, g: 92, b: 246 };

function cellColor(val: number, nMax: number): { bg: string; fg: string } {
  if (nMax <= 0) return { bg: "rgb(255,255,255)", fg: "#1a1a1a" };
  const t = Math.min(1, val / nMax);
  const r = Math.round(255 + (PURPLE.r - 255) * t);
  const g = Math.round(255 + (PURPLE.g - 255) * t);
  const b = Math.round(255 + (PURPLE.b - 255) * t);
  return {
    bg: `rgb(${r},${g},${b})`,
    fg: t > 0.45 ? "#fff" : "#1a1a1a",
  };
}

export function ConfusionMatrixChart({ matrix, labels, nMax }: ConfusionMatrixChartProps) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const { width } = useChartSize(containerRef);

  const numCols = labels.length;
  if (!numCols || !matrix.length) return null;

  // nMax = maximum test images for any class (constant across same test set)
  const scaleMax = useMemo(() => {
    if (nMax !== undefined) return nMax;
    return Math.max(...matrix.map((row) => row.reduce((s, v) => s + v, 0)));
  }, [matrix, nMax]);

  // Responsive sizes
  const isCompact = width < 600;
  const labelW = isCompact ? 70 : 160;
  const cellSize = isCompact ? 40 : Math.min(80, Math.max(44, Math.floor((width - labelW - 32) / numCols)));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{t("experiment_detail.charts.confusion_title")}</CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={containerRef} className="overflow-x-auto">
          {/* Column headers */}
          <div className="flex mb-[1px]" style={{ paddingLeft: labelW + 8 }}>
            {labels.map((label) => (
              <div
                key={label}
                className="text-xs text-muted-foreground text-center truncate leading-tight"
                style={{ width: cellSize, minWidth: cellSize, paddingBottom: 4 }}
                title={label}
              >
                {label.replace("O. ", "")}
              </div>
            ))}
            {/* right padding for row totals */}
            <div className="text-xs text-muted-foreground text-center" style={{ width: cellSize, minWidth: cellSize, paddingBottom: 4 }}>∑</div>
          </div>

          {/* Rows */}
          {matrix.map((row, i) => {
            const rowSum = row.reduce((s, v) => s + v, 0);
            return (
              <div key={i} className="flex items-stretch mb-[1px]">
                {/* Row label */}
                <div
                  className="text-xs text-muted-foreground text-right pr-2 truncate flex items-center justify-end"
                  style={{ width: labelW, minWidth: labelW, height: cellSize }}
                  title={labels[i]}
                >
                  {labels[i].replace("O. ", "")}
                </div>

                {/* Cells */}
                {row.map((val, j) => {
                  const { bg, fg } = cellColor(val, scaleMax);
                  const isDiag = i === j;
                  return (
                    <div
                      key={j}
                      className="flex items-center justify-center text-xs font-mono border border-border/40 transition-colors duration-100"
                      style={{
                        width: cellSize,
                        minWidth: cellSize,
                        height: cellSize,
                        backgroundColor: bg,
                        color: fg,
                        fontWeight: isDiag && val > 0 ? 700 : 400,
                        borderColor: isDiag ? `rgba(${PURPLE.r},${PURPLE.g},${PURPLE.b},0.5)` : undefined,
                      }}
                      title={`${labels[i]} → ${labels[j]}: ${val}`}
                    >
                      {val}
                    </div>
                  );
                })}

                {/* Row total */}
                <div
                  className="flex items-center justify-center text-xs font-mono text-muted-foreground border border-transparent"
                  style={{ width: cellSize, minWidth: cellSize, height: cellSize }}
                >
                  {rowSum}
                </div>
              </div>
            );
          })}

          {/* Bottom legend: gradient bar */}
          <div className="flex items-center gap-3 mt-4 text-xs text-muted-foreground">
            <span>0</span>
            <div
              className="flex-1 rounded-sm"
              style={{
                height: 10,
                background: `linear-gradient(to right, rgb(255,255,255) 0%, rgb(${PURPLE.r},${PURPLE.g},${PURPLE.b}) 100%)`,
              }}
            />
            <span>{scaleMax}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
