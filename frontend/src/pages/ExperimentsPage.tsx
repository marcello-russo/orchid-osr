// frontend/src/pages/ExperimentsPage.tsx

import { useState } from "react";
import { Link } from "react-router-dom";
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
  const { data: experiments, isLoading } = useExperiments();
  const [filter, setFilter] = useState<"all" | "kfold" | "baseline">("all");

  const filtered = (experiments ?? []).filter(
    (e) => filter === "all" || e.type === filter
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-primary">🧪 Experiments</h1>
        <div className="flex gap-2">
          {(["all", "kfold", "baseline"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={cn(
                "rounded-lg px-4 py-2 text-sm font-medium transition-colors duration-150 min-h-[44px]",
                filter === f
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground bg-muted/50 hover:bg-muted"
              )}
            >
              {f === "all" ? "All" : f === "kfold" ? "K-Fold" : "Baseline"}
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
            {filter !== "all" ? `No ${filter} experiments found.` : "No experiments found."}
          </p>
        </div>
      ) : (
        <div className="rounded-lg border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="sticky left-0 bg-card z-10 shadow-[2px_0_4px_-2px_rgba(0,0,0,0.08)]">Name</TableHead>
                <TableHead className="whitespace-nowrap">Model</TableHead>
                <TableHead>Type</TableHead>
                <TableHead className="text-right">Mean F1</TableHead>
                <TableHead className="text-right">Test Acc</TableHead>
                <TableHead className="text-right whitespace-nowrap">Folds</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((exp) => (
                <TableRow key={exp.id} className="transition-colors duration-150">
                  <TableCell className="sticky left-0 bg-card z-10 shadow-[2px_0_4px_-2px_rgba(0,0,0,0.08)]">
                    <Link
                      to={`/experiments/${exp.id}`}
                      className="font-medium text-primary hover:underline"
                    >
                      {exp.name}
                    </Link>
                  </TableCell>
                  <TableCell className="font-mono text-sm whitespace-nowrap">
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
                  <TableCell className="text-right whitespace-nowrap">
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
