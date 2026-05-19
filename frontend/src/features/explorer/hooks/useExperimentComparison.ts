import { useQuery } from "@tanstack/react-query";
import { fetchExperimentComparison } from "@/lib/api";
import type { ExperimentComparisonResponse } from "@/lib/types";

export function useExperimentComparison() {
  return useQuery<ExperimentComparisonResponse>({
    queryKey: ["comparison", "experiments"],
    queryFn: fetchExperimentComparison,
    staleTime: 5 * 60 * 1000,
  });
}
