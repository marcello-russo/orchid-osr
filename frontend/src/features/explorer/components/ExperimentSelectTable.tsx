import { useState } from "react";
import { useExperiments } from "@/features/explorer/hooks/useExperiments";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

interface Props {
  currentId: string;
  onCompare: (ids: string[]) => void;
}

export function ExperimentSelectTable({ currentId, onCompare }: Props) {
  const { data: allExperiments, isLoading } = useExperiments();
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const available = (allExperiments ?? []).filter((e) => e.id !== currentId);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < 3) next.add(id);
      return next;
    });
  };

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => <Skeleton key={i} className="h-10 w-full rounded-lg" />)}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10"></TableHead>
              <TableHead>Experiment</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Type</TableHead>
              <TableHead className="text-right">F1</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {available.map((exp) => {
              const checked = selected.has(exp.id);
              return (
                <TableRow key={exp.id} className="cursor-pointer" onClick={() => toggle(exp.id)}>
                  <TableCell>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggle(exp.id)}
                      disabled={!checked && selected.size >= 3}
                      className="accent-primary size-4 cursor-pointer"
                      onClick={(e) => e.stopPropagation()}
                    />
                  </TableCell>
                  <TableCell className="font-medium">{exp.id}</TableCell>
                  <TableCell className="font-mono text-sm">{exp.model_variant ?? "—"}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className="text-xs">{exp.type}</Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {exp.mean_f1 != null ? `${(exp.mean_f1 * 100).toFixed(2)}%` : "—"}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
      <div className="flex justify-end">
        <Button onClick={() => onCompare(Array.from(selected))} disabled={selected.size === 0}>
          Compare ({selected.size})
        </Button>
      </div>
    </div>
  );
}
