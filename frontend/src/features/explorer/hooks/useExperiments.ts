// frontend/src/features/explorer/hooks/useExperiments.ts

import { useQuery } from "@tanstack/react-query";
import { fetchExperiments } from "@/lib/api";
import type { ExperimentSummary } from "@/lib/types";

export function useExperiments() {
  return useQuery<ExperimentSummary[]>({
    queryKey: ["experiments"],
    queryFn: fetchExperiments,
  });
}
