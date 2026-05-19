import { useQueries } from "@tanstack/react-query";
import { fetchExperiment } from "@/lib/api";
import type { ExperimentDetail } from "@/lib/types";

export function useCompareExperiments(ids: string[]) {
  const queries = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["experiment", id],
      queryFn: () => fetchExperiment(id),
      enabled: !!id,
    })),
  });

  const experiments: ExperimentDetail[] = queries
    .filter((q) => q.data !== undefined)
    .map((q) => q.data!);

  const isLoading = queries.some((q) => q.isLoading);
  const errors = queries.filter((q) => q.error).map((q) => q.error);

  return { experiments, isLoading, errors };
}
