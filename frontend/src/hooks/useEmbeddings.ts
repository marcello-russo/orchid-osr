import { useQuery } from "@tanstack/react-query";
import { fetchEmbeddings, fetchPublicEmbeddings } from "@/lib/api";
import type { EmbeddingResponse } from "@/lib/types";

export function useEmbeddings(experiment?: string, method: string = "tsne_3d", enabled = true) {
  return useQuery<EmbeddingResponse>({
    queryKey: ["embeddings", experiment, method],
    queryFn: () => fetchEmbeddings(experiment, method),
    staleTime: 5 * 60 * 1000,
    retry: false,
    enabled,
  });
}

/** Vetrina pubblica: esperimento fisso, coordinate precalcolate, nessuna immagine. */
export function usePublicEmbeddings() {
  return useQuery<EmbeddingResponse>({
    queryKey: ["embeddings", "public"],
    queryFn: fetchPublicEmbeddings,
    staleTime: 60 * 60 * 1000,
    retry: false,
  });
}
