import { useMutation } from "@tanstack/react-query";
import { classifyUpload } from "@/lib/api";
import type { ClassifyUploadResponse } from "@/lib/types";
import { useState } from "react";

export function useClassify() {
  const [progress, setProgress] = useState(0);

  const mutation = useMutation<ClassifyUploadResponse, Error, { file: File; modelId?: string }>({
    mutationFn: ({ file, modelId }) =>
      classifyUpload(file, modelId, (p: number) => setProgress(p)),
    onSettled: () => setProgress(0),
  });

  return { ...mutation, progress };
}
