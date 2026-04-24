import { useMutation } from "@tanstack/react-query";
import { explainImage } from "@/lib/api";
import type { ExplainResponse } from "@/lib/types";
import { useState } from "react";

export function useExplain() {
  const [progress, setProgress] = useState(0);

  const mutation = useMutation<ExplainResponse, Error, { file: File; modelId?: string }>({
    mutationFn: ({ file, modelId }) =>
      explainImage(file, modelId, (p: number) => setProgress(p)),
    onSettled: () => setProgress(0),
  });

  return { ...mutation, progress };
}
