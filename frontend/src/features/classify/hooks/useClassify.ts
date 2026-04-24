import { useMutation } from "@tanstack/react-query";
import { classifyUpload } from "@/lib/api";
import type { ClassifyUploadResponse } from "@/lib/types";

export function useClassify() {
  return useMutation<ClassifyUploadResponse, Error, { file: File; modelId?: string }>({
    mutationFn: ({ file, modelId }) => classifyUpload(file, modelId),
  });
}
