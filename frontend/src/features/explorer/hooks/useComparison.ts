import { useQuery } from "@tanstack/react-query";
import { fetchComparison } from "@/lib/api";
import type { ModelComparisonResponse } from "@/lib/types";

export function useComparison() {
  return useQuery<ModelComparisonResponse>({
    queryKey: ["comparison"],
    queryFn: fetchComparison,
    staleTime: 5 * 60 * 1000,
  });
}
