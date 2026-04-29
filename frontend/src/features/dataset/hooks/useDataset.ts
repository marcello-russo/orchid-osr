// frontend/src/features/dataset/hooks/useDataset.ts
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { fetchDatasetClasses, fetchClassImages } from "@/lib/api";
import type { DatasetClassesResponse, ClassImagesResponse } from "@/lib/types";

export function useDatasetClasses() {
  return useQuery<DatasetClassesResponse>({
    queryKey: ["dataset", "classes"],
    queryFn: fetchDatasetClasses,
  });
}

export function useClassImages(
  className: string,
  split: string = "train",
  limit: number = 12,
  offset: number = 0
) {
  return useQuery<ClassImagesResponse>({
    queryKey: ["dataset", "classes", className, "images", split, limit, offset],
    queryFn: () => fetchClassImages(className, split, limit, offset),
    enabled: !!className,
    placeholderData: keepPreviousData,
  });
}
