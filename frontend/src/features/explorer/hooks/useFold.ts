// frontend/src/features/explorer/hooks/useFold.ts

import { useQuery } from "@tanstack/react-query";
import { fetchFold } from "@/lib/api";
import type { FoldData } from "@/lib/types";

export function useFold(experimentId: string, foldNumber: number) {
  return useQuery<FoldData>({
    queryKey: ["experiment", experimentId, "fold", foldNumber],
    queryFn: () => fetchFold(experimentId, foldNumber),
    enabled: !!experimentId && foldNumber >= 0,
  });
}
