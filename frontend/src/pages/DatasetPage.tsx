import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Image as ImageIcon } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDatasetClasses } from "@/features/dataset/hooks/useDataset";
import { ClassGallery } from "@/features/dataset/components/ClassGallery";
import { DistributionChart } from "@/features/dataset/components/DistributionChart";

export function DatasetPage() {
  const { t } = useTranslation();
  const { data, isLoading, error } = useDatasetClasses();
  const [split, setSplit] = useState<"train_clean" | "test_clean">("train_clean");

  if (isLoading) {
    return (
      <div className="space-y-4 animate-in fade-in duration-300">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full rounded-lg" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="py-20 text-center text-muted-foreground">
        <p>{t("common.error")}</p>
        <p className="text-sm mt-1">{error?.message}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-primary flex items-center gap-2">
          <ImageIcon className="size-6" />
          {t("dataset.title")}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {data.total_train} {t("dataset.tabs.train")} / {data.total_test} {t("dataset.tabs.test")} / {data.classes.length} classes
        </p>
      </div>

      {/* Distribution chart */}
      <DistributionChart classes={data.classes} />

      {/* Split toggle */}
      <Tabs value={split} onValueChange={(v) => setSplit(v as "train_clean" | "test_clean")}>
        <TabsList>
          <TabsTrigger value="train_clean">{t("dataset.tabs.train")} ({data.total_train})</TabsTrigger>
          <TabsTrigger value="test_clean">{t("dataset.tabs.test")} ({data.total_test})</TabsTrigger>
        </TabsList>
      </Tabs>

      {/* Gallery per class */}
      <div className="space-y-4">
        {data.classes.map((cls) => (
          <ClassGallery key={cls.name} classInfo={cls} split={split} totalImages={data.total_train + data.total_test} />
        ))}
      </div>
    </div>
  );
}
