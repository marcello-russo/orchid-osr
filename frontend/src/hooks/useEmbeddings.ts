import { useQuery } from "@tanstack/react-query";
import { fetchEmbeddings } from "@/lib/api";
import type { EmbeddingResponse } from "@/lib/types";

export function useEmbeddings(experiment?: string, method: string = "tsne_3d") {
  return useQuery<EmbeddingResponse>({
    queryKey: ["embeddings", experiment, method],
    queryFn: () => fetchEmbeddings(experiment, method),
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
}
