// frontend/src/features/explorer/hooks/useExperiment.ts

import { useQuery } from "@tanstack/react-query";
import { fetchExperiment } from "@/lib/api";
import type { ExperimentDetail } from "@/lib/types";

export function useExperiment(id: string) {
  return useQuery<ExperimentDetail>({
    queryKey: ["experiment", id],
    queryFn: () => fetchExperiment(id),
    enabled: !!id,
  });
}
