import { useMutation } from "@tanstack/react-query";
import { explainImage } from "@/lib/api";
import type { ExplainResponse } from "@/lib/types";

export function useExplain() {
  return useMutation<ExplainResponse, Error, { file: File; modelId?: string }>({
    mutationFn: ({ file, modelId }) => explainImage(file, modelId),
  });
}
