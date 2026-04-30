import { useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { FlaskConical } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { useExperiments } from "@/features/explorer/hooks/useExperiments";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export function ExperimentsPage() {
  const { t } = useTranslation();
  const { data: experiments, isLoading } = useExperiments();
  const [filter, setFilter] = useState<"all" | "kfold" | "baseline">("all");

  const filtered = (experiments ?? []).filter(
    (e) => filter === "all" || e.type === filter
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-2xl font-bold text-primary">🧪 {t("experiments.title")}</h1>
        <div className="flex flex-wrap gap-2">
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
          <p className="text-sm">
            {t("common.error")}
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("experiments.table.experiment")}</TableHead>
                <TableHead>{t("experiments.table.model")}</TableHead>
                <TableHead>Type</TableHead>
                <TableHead className="text-right">{t("experiments.table.f1")}</TableHead>
                <TableHead className="text-right">Test Acc</TableHead>
                <TableHead className="text-right">Folds</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((exp) => (
                <TableRow key={exp.id} className="transition-colors duration-150">
                  <TableCell className="font-medium">
                    <Link
                      to={`/experiments/${exp.id}`}
                      className="text-primary hover:underline"
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
                        exp.type === "kfold" &&
                          "bg-primary/10 text-primary"
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
                  <TableCell className="text-right">
                    {exp.num_folds ?? "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
