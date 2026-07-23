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

export function MatrixSection({
  lookup, architectureRows, columns, title, accessor, colorFn, bestPerColumn, tooltipLabel,
}: MatrixSectionProps) {
  const navigate = useNavigate();
  const { t } = useTranslation();

  const renderCell = (col: string, variant: string) => {
    const exp = lookup[variant]?.[col];
    const value = exp ? accessor(exp) : null;
    const isBest = value !== null && value === bestPerColumn[col];
    const formattedValue = value !== null ? `${(value * 100).toFixed(1)}` : "—";
    const tooltipId = exp ? `tooltip-${variant}-${col}` : undefined;

    return (
      <td
        key={col}
        className={cn(
          "group relative text-center py-2 px-3 rounded-lg transition-all",
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
        aria-describedby={tooltipId}
      >
        <span>{formattedValue}</span>
        {isBest && <span className="ml-0.5 text-yellow-300" aria-label="best performing architecture">★</span>}
        {exp && (
          <div id={tooltipId} className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block group-focus-within:block z-50">
            <div className="bg-popover text-popover-foreground text-xs rounded-lg border border-border px-3 py-2 shadow-md whitespace-nowrap">
              <p className="font-semibold">{exp.name}</p>
              <p>{tooltipLabel}: {formattedValue}%</p>
              {value !== null && exp.std_f1 != null && (
                <p>Std: ±{(exp.std_f1 * 100).toFixed(2)}%</p>
              )}
              {exp.test_accuracy != null && (
                <p>Test Acc: {(exp.test_accuracy * 100).toFixed(2)}%</p>
              )}
            </div>
          </div>
        )}
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
      <div className="overflow-x-auto">
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
    </div>
  );
}
