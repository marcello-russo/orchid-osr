import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Database } from "lucide-react";
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
    <div className="space-y-6 max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="size-10 md:size-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center text-emerald-500">
          <Database className="size-6" />
        </div>
        <div>
          <h1 className="text-2xl md:text-4xl font-black tracking-tight">{t("dataset.title")}</h1>
          <p className="text-muted-foreground text-sm mt-1">{t("dataset.subtitle")}</p>
        </div>
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
