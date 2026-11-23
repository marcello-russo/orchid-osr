import { TableHead } from "@/components/ui/table";
import type { SortDir } from "@/lib/sort";
import { cn } from "@/lib/utils";

interface SortHeaderProps {
  label: string;
  sortKey: string;
  activeKey: string;
  dir: SortDir;
  onToggle: (key: string) => void;
  align?: "left" | "right";
  className?: string;
}

export function SortHeader({
  label, sortKey, activeKey, dir, onToggle, align = "left", className,
}: SortHeaderProps) {
  const isActive = sortKey === activeKey;
  return (
    <TableHead
      className={cn(
        "cursor-pointer select-none",
        align === "right" && "text-right",
        className,
      )}
      onClick={() => onToggle(sortKey)}
      aria-sort={isActive ? (dir === "asc" ? "ascending" : "descending") : undefined}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {isActive && (
          <span className="text-xs text-muted-foreground" aria-hidden="true">
            {dir === "asc" ? "↑" : "↓"}
          </span>
        )}
      </span>
    </TableHead>
  );
}
