import { useTranslation } from "react-i18next";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface Column {
  key: string;
  label: string;
}

interface DataTableProps {
  columns: Column[];
  rows: Record<string, string | number>[];
  /** Key to use as row identifier (default: first column key) */
  rowKey?: string;
  /** Highlight the entire row whose rowKey matches this value */
  highlightRow?: string;
  /** Per-column best-value highlighting: map columnKey → rowKey value to mark as best */
  bests?: Record<string, string>;
  caption?: string;
}

export function DataTable({ columns, rows, rowKey, highlightRow, bests, caption }: DataTableProps) {
  const { t } = useTranslation();
  const keyAccessor = rowKey ?? columns[0].key;

  return (
    <div className="overflow-x-auto rounded-lg border border-border my-6">
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((col) => (
              <TableHead key={col.key} className="whitespace-nowrap">{col.label}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => {
            const rowId = String(row[keyAccessor]);
            const isRowHighlighted = highlightRow === rowId;
            return (
              <TableRow
                key={rowId}
                className={cn(
                  isRowHighlighted && "bg-primary/5",
                )}
              >
                {columns.map((col) => {
                  const isBestInCol = bests?.[col.key] === rowId;
                  return (
                    <TableCell
                      key={col.key}
                      className={cn(
                        "transition-colors",
                        isRowHighlighted && "font-medium",
                        isBestInCol && "bg-amber-50 dark:bg-amber-950/40 font-bold text-foreground",
                      )}
                    >
                      <span className={cn(
                        "inline-flex items-center gap-1",
                        isBestInCol && "text-foreground",
                      )}>
                        {isBestInCol && <span className="text-amber-500 text-xs" aria-label={t("research.datatable_best_value")}>★</span>}
                        {row[col.key]}
                      </span>
                    </TableCell>
                  );
                })}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      {caption && <p className="text-xs text-muted-foreground px-3 pb-3 italic">{caption}</p>}
    </div>
  );
}
