import { useState, useCallback, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import type { ExperimentSummary } from "@/lib/types";
import {
  getArchitectureFamily, FAMILY_COLORS, LABEL_TO_VARIANT,
} from "@/features/explorer/types";

interface MatrixSectionProps {
  lookup: Record<string, Record<string, ExperimentSummary>>;
  architectureRows: string[];
  columns: string[];
  title?: string;
  accessor: (exp: ExperimentSummary) => number | null;
  colorFn: (value: number) => string;
  bestPerColumn: Record<string, number>;
  tooltipLabel: string;
}

interface TooltipState {
  exp: ExperimentSummary;
  formattedValue: string;
  value: number | null;
  top: number;
  left: number;
}

export function MatrixSection({
  lookup, architectureRows, columns, title, accessor, colorFn, bestPerColumn, tooltipLabel,
}: MatrixSectionProps) {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout>>(null);

  useEffect(() => {
    return () => {
      if (closeTimer.current) clearTimeout(closeTimer.current);
    };
  }, []);

  const handleMouseEnter = useCallback((
    e: React.MouseEvent<HTMLTableCellElement>,
    exp: ExperimentSummary,
    formattedValue: string,
    value: number | null,
  ) => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    const rect = e.currentTarget.getBoundingClientRect();
    setTooltip({ exp, formattedValue, value, top: rect.top - 8, left: rect.left + rect.width / 2 });
  }, []);

  const handleMouseLeave = useCallback(() => {
    closeTimer.current = setTimeout(() => setTooltip(null), 80);
  }, []);

  const renderCell = (col: string, variant: string) => {
    const exp = lookup[variant]?.[col];
    const value = exp ? accessor(exp) : null;
    const isBest = value !== null && value === bestPerColumn[col];
    const formattedValue = value !== null ? `${(value * 100).toFixed(1)}%` : "—";
    const stdValue = exp?.std_f1 != null ? ` ± ${(exp.std_f1 * 100).toFixed(1)}%` : "";

    return (
      <td
        key={col}
        className={cn(
          "relative text-center py-2 px-3 rounded-lg transition-all",
          exp
            ? "cursor-pointer focus-visible:ring-2 focus-visible:ring-ring hover:brightness-110"
            : "text-muted-foreground",
        )}
        style={{ backgroundColor: value !== null ? colorFn(value) : undefined }}
        onClick={() => exp && navigate(`/experiments/${exp.id}`)}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && exp) {
            e.preventDefault();
            navigate(`/experiments/${exp.id}`);
          }
        }}
        tabIndex={exp ? 0 : undefined}
        role={exp ? "button" : undefined}
        aria-label={exp ? `View details: ${exp.name}` : undefined}
        onMouseEnter={(e) => exp && handleMouseEnter(e, exp, formattedValue, value)}
        onMouseLeave={handleMouseLeave}
      >
        <div className="flex flex-col items-center leading-tight">
          <span>{formattedValue}</span>
          {stdValue && <span className="text-[10px] opacity-70">{stdValue}</span>}
        </div>
        {isBest && <span className="ml-0.5 text-yellow-300" aria-label="best performing architecture">★</span>}
      </td>
    );
  };

  return (
    <div>
      {title && (
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          {title}
        </h3>
      )}
      <div className="overflow-x-auto w-full">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left py-2 pr-4 font-medium text-muted-foreground text-xs uppercase tracking-wider">
                {t("experiments.table.model")}
              </th>
              {columns.map((s) => (
                <th key={s} className="text-center py-2 px-3 font-medium text-xs uppercase tracking-wider text-muted-foreground">
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border/20">
            {architectureRows.map((archLabel) => {
              const variant = LABEL_TO_VARIANT[archLabel];
              const family = variant ? getArchitectureFamily(variant) : null;
              return (
                <tr key={archLabel} className="hover:bg-muted/20 transition-colors">
                  <td className="py-2 pr-4 font-medium text-sm flex items-center gap-2">
                    <span
                      className="inline-block size-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: family ? FAMILY_COLORS[family] : "gray" }}
                    />
                    <span>{archLabel}</span>
                  </td>
                  {columns.map((col) => renderCell(col, variant))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {tooltip && createPortal(
        <div
          className="fixed z-50"
          style={{ top: tooltip.top, left: tooltip.left, transform: "translate(-50%, -100%)" }}
          onMouseEnter={() => { if (closeTimer.current) clearTimeout(closeTimer.current); }}
        >
          <div className="bg-popover text-popover-foreground text-xs rounded-lg border border-border px-3 py-2 shadow-md whitespace-nowrap">
            <p className="font-semibold">{tooltip.exp.name}</p>
            <p>{tooltipLabel}: {tooltip.formattedValue}</p>
            {tooltip.value !== null && tooltip.exp.std_f1 != null && (
              <p>Std: ±{(tooltip.exp.std_f1 * 100).toFixed(2)}%</p>
            )}
            {tooltip.exp.test_accuracy != null && (
              <p>Test Acc: {(tooltip.exp.test_accuracy * 100).toFixed(2)}%</p>
            )}
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}
