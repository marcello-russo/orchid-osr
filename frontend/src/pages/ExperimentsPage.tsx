import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { FlaskConical, ArrowRight } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useExperiments } from "@/features/explorer/hooks/useExperiments";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export function ExperimentsPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { data: experiments, isLoading } = useExperiments();
  const [filter, setFilter] = useState<"all" | "kfold" | "baseline">("all");
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const filtered = (experiments ?? []).filter(
    (e) => filter === "all" || e.type === filter
  );

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < 4) next.add(id);
      return next;
    });
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-purple-500/10 flex items-center justify-center text-purple-500">
            <FlaskConical className="size-6" />
          </div>
          <div>
            <h1 className="text-2xl md:text-4xl font-black tracking-tight">{t("experiments.title")}</h1>
            <p className="text-muted-foreground text-sm mt-1">{t("experiments.subtitle")}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {(["all", "kfold", "baseline"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={cn(
                "rounded-lg px-4 py-2 text-sm font-medium transition-colors duration-150 cursor-pointer",
                filter === f
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground bg-muted/50 hover:bg-muted"
              )}
            >
              {t(`experiments.filters.${f}`)}
            </button>
          ))}
          {selected.size >= 2 && (
            <Button
              size="sm"
              onClick={() => {
                navigate(`/compare?ids=${Array.from(selected).join(",")}`);
              }}
            >
              Compare ({selected.size}) <ArrowRight className="ml-1 size-4" />
            </Button>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-12 w-full rounded-lg" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
          <FlaskConical className="mb-3 size-10 opacity-30" />
          <p className="text-sm">{t("common.error")}</p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10">
                  <span className="sr-only">Select</span>
                </TableHead>
                <TableHead>{t("experiments.table.experiment")}</TableHead>
                <TableHead>{t("experiments.table.model")}</TableHead>
                <TableHead>{t("experiments.table.type")}</TableHead>
                <TableHead className="text-right">{t("experiments.table.f1")}</TableHead>
                <TableHead className="text-right">{t("experiments.table.test_acc")}</TableHead>
                <TableHead className="text-right">{t("experiments.table.folds")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((exp) => {
                const isSelected = selected.has(exp.id);
                return (
                  <TableRow
                    key={exp.id}
                    className={cn(
                      "transition-colors duration-150 cursor-pointer",
                      isSelected && "bg-primary/5"
                    )}
                    onClick={() => toggle(exp.id)}
                  >
                    <TableCell className="w-10">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggle(exp.id)}
                        className="accent-primary size-4 cursor-pointer"
                        onClick={(e) => e.stopPropagation()}
                      />
                    </TableCell>
                    <TableCell className="font-medium">
                      <Link
                        to={`/experiments/${exp.id}`}
                        className="text-primary hover:underline"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {exp.name}
                      </Link>
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {exp.model_variant ?? "—"}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={exp.type === "kfold" ? "default" : "secondary"}
                        className={cn(
                          "text-xs",
                          exp.type === "kfold" && "bg-primary/10 text-primary"
                        )}
                      >
                        {exp.type}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {exp.mean_f1 !== null
                        ? `${(exp.mean_f1 * 100).toFixed(2)}%`
                        : "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {exp.test_accuracy !== null
                        ? `${(exp.test_accuracy * 100).toFixed(2)}%`
                        : "—"}
                    </TableCell>
                    <TableCell className="text-right">{exp.folds_completed != null ? `${exp.folds_completed}/${exp.num_folds ?? "?"}` : "—"}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
