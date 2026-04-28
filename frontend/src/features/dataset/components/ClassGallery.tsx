// frontend/src/features/dataset/components/ClassGallery.tsx
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useClassImages } from "@/features/dataset/hooks/useDataset";
import type { DatasetClassInfo } from "@/lib/types";

const PAGE_SIZE = 12;

interface ClassGalleryProps {
  classInfo: DatasetClassInfo;
  split: "train_clean" | "test_clean";
}

export function ClassGallery({ classInfo, split }: ClassGalleryProps) {
  const [offset, setOffset] = useState(0);
  useEffect(() => { setOffset(0); }, [split]);
  const apiSplit = split === "train_clean" ? "train" : "test";
  const { data, isLoading } = useClassImages(classInfo.name, apiSplit, PAGE_SIZE, offset);
  const hasMore = data ? offset + PAGE_SIZE < data.total : false;

  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{classInfo.name}</CardTitle>
          <span className="text-sm text-muted-foreground">
            {classInfo.total} images ({classInfo.train_count} train / {classInfo.test_count} test)
          </span>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        {isLoading && !data ? (
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
            {Array.from({ length: PAGE_SIZE }).map((_, i) => (
              <Skeleton key={i} className="aspect-square rounded-md" />
            ))}
          </div>
        ) : data && data.images.length > 0 ? (
          <>
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
              {data.images.map((img) => (
                <div key={img.filename} className="group relative aspect-square overflow-hidden rounded-md bg-muted">
                  <img
                    src={img.url}
                    alt={img.filename}
                    loading="lazy"
                    className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105"
                  />
                </div>
              ))}
            </div>
            {hasMore && (
              <div className="mt-3 text-center">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setOffset((o) => o + PAGE_SIZE)}
                >
                  Show more ({data.total - offset - PAGE_SIZE} remaining)
                </Button>
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">No images found.</p>
        )}
      </CardContent>
    </Card>
  );
}
