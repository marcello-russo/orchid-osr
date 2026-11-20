// frontend/src/features/dataset/components/ClassGallery.tsx
import { useState, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useClassImages } from "@/features/dataset/hooks/useDataset";
import type { DatasetClassInfo } from "@/lib/types";

const PAGE_SIZE = 8;

interface ClassGalleryProps {
  classInfo: DatasetClassInfo;
  split: "train" | "test";
  totalImages: number;
}

import { useInView } from "@/hooks/useInView";
import { useAppStore } from "@/stores/useAppStore";

export function ClassGallery({ classInfo, split, totalImages }: ClassGalleryProps) {
  const { t } = useTranslation();
  const [offset, setOffset] = useState(0);
  const { ref: cardRef, isInView } = useInView({ rootMargin: "200px" });
  
  useEffect(() => { setOffset(0); }, [split]);

  const total = split === "train" ? classInfo.train_count : classInfo.test_count;
  const totalPages = Math.ceil(total / PAGE_SIZE);
  // Skip API call if this split has 0 images for this class
  const skipQuery = total === 0;

  const { data, isLoading, isPlaceholderData } = useClassImages(
    classInfo.name, 
    split, 
    PAGE_SIZE, 
    offset,
    isInView && !skipQuery // Pass isInView to enable the query, skip if 0 images
  );

  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;
  const hasMore = offset + PAGE_SIZE < total;
  const isFetchingNext = isLoading || isPlaceholderData;

  return (
    <Card ref={cardRef}>
      <CardHeader className="pb-2 border-b border-border/50">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <CardTitle className="text-base">{classInfo.name}</CardTitle>
          <div className="flex items-center gap-4">
            <span className="text-xs text-muted-foreground whitespace-nowrap">
              {classInfo.total} {t("dataset.images")} ({classInfo.train_count} {t("dataset.train_label")} / {classInfo.test_count} {t("dataset.test_label")})
            </span>
            {totalPages > 1 && (
              <div className="flex items-center gap-1">
                <Button
                  variant="outline"
                  size="icon"
                  className="size-7"
                  disabled={offset === 0 || isFetchingNext}
                  onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                >
                  <ChevronLeft className="size-4" />
                </Button>
                <span className="text-[10px] font-medium min-w-[3rem] text-center">
                  {currentPage} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="icon"
                  className="size-7"
                  disabled={!hasMore || isFetchingNext}
                  onClick={() => setOffset((o) => o + PAGE_SIZE)}
                >
                  <ChevronRight className="size-4" />
                </Button>
              </div>
            )}
          </div>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <div className="h-1 flex-1 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-primary/60 transition-all"
              style={{ width: `${(classInfo.total / totalImages) * 100}%` }}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        {isLoading ? (
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
            {Array.from({ length: PAGE_SIZE }).map((_, i) => (
              <Skeleton key={i} className="aspect-[1/2] rounded-md" />
            ))}
          </div>
        ) : data && data.images.length > 0 ? (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 animate-in fade-in zoom-in-95 duration-300">
            {data.images.map((img) => (
                <div key={img.filename} className="group relative aspect-[1/2] overflow-hidden rounded-md bg-muted/50">
                <ImageWithAuth
                  imageUrl={`${img.url}?size=300`}
                  token={useAppStore.getState().token ?? ""}
                  alt={img.filename}
                />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">{t("dataset.no_images")}</p>
        )}
      </CardContent>
    </Card>
  );
}

function ImageWithAuth({ imageUrl, token, alt }: { imageUrl: string; token: string; alt: string }) {
  const [src, setSrc] = useState<string>("");
  const [isLoaded, setIsLoaded] = useState(false);
  const blobUrlRef = useRef<string | null>(null);
  useEffect(() => {
    // Senza token si scarica lo stesso: il backend serve l'anteprima pubblica
    // (3 foto per classe) e rifiuta il resto. Prima qui c'era un `return` che
    // lasciava la galleria pubblica vuota.
    let cancelled = false;
    const init = token ? { headers: { Authorization: `Bearer ${token}` } } : undefined;
    fetch(imageUrl, init)
      .then(r => r.blob())
      .then(blob => {
        if (cancelled) return;
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = URL.createObjectURL(blob);
        setSrc(blobUrlRef.current);
      });
    return () => { cancelled = true; };
  }, [imageUrl, token]);

  useEffect(() => {
    return () => { if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current); };
  }, []);

  return (
    <div className="relative h-full w-full">
      {!src && <Skeleton className="h-full w-full rounded-md" />}
      {src && !isLoaded && (
        <>
          <Skeleton className="absolute inset-0 h-full w-full rounded-md" />
          <img
            src={src}
            alt={alt}
            onLoad={() => setIsLoaded(true)}
            className="absolute inset-0 h-full w-full object-cover opacity-0 scale-95"
          />
        </>
      )}
      {src && isLoaded && (
        <img
          src={src}
          alt={alt}
          className="h-full w-full object-cover transition-all duration-500 group-hover:scale-105 opacity-100 scale-100"
        />
      )}
    </div>
  );
}
