import { useState, useRef, useCallback } from "react";
import { useTranslation } from "react-i18next";
import {
  Upload, Loader2, Sparkles, Eye, X, ScanSearch,
  Download, Trash2, HardDrive, Cloud, RefreshCw,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Card, CardContent, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useClassify } from "@/features/classify/hooks/useClassify";
import { useExplain } from "@/features/classify/hooks/useExplain";
import { useModels } from "@/features/classify/hooks/useModels";
import {
  fetchCacheStatus, downloadAllModels, downloadModel, clearModelCache,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { InfoTooltip } from "@/components/ui/InfoTooltip";

export function ClassifyPage() {
  const { t } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const classify = useClassify();
  const explain = useExplain();
  const { data: modelsData, isLoading, isError, error } = useModels();
  const models = modelsData?.models ?? [];

  const cacheQuery = useQuery({
    queryKey: ["cache-status"],
    queryFn: fetchCacheStatus,
    refetchInterval: (q) => q.state.data?.is_downloading ? 2000 : false,
  });

  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Auto-select first model once models load
  if (!selectedModelId && models.length > 0) {
    setSelectedModelId(models[0].id);
  }

  const isWorking = classify.isPending || explain.isPending;
  const activeProgress = classify.isPending
    ? classify.progress : explain.isPending ? explain.progress : 0;
  const uploadDone = isWorking && activeProgress >= 100;
  const activeLabel = uploadDone
    ? t("common.loading") : classify.isPending
      ? t("classify.progress.uploading") : t("classify.progress.uploading");

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
    setHeatmapUrl(null);
    setShowHeatmap(false);
    classify.reset();
    explain.reset();
  }, [classify, explain]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const handleClassify = () => {
    if (!file) return;
    classify.mutate({ file, modelId: selectedModelId ?? undefined });
  };

  const handleExplain = () => {
    if (!file) return;
    explain.mutate(
      { file, modelId: selectedModelId ?? undefined },
      { onSuccess: (data) => {
        setHeatmapUrl(`data:image/png;base64,${data.heatmap_base64}`);
        setShowHeatmap(true);
      }},
    );
  };

  const clearAll = () => {
    setFile(null);
    setPreviewUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; });
    setHeatmapUrl(null);
    setShowHeatmap(false);
    classify.reset();
    explain.reset();
    if (inputRef.current) inputRef.current.value = "";
  };

  const topPredictions = classify.data?.top_predictions ?? [];
  const cache = cacheQuery.data;

  // Group models by experiment
  const grouped: Record<string, typeof models> = {};
  models.forEach((m) => {
    const key = m.exp_label || `Exp ${m.exp_num ?? "?"}`;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(m);
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary">
            <ScanSearch className="size-6" />
          </div>
          <div>
            <h1 className="text-2xl md:text-4xl font-black tracking-tight">{t("classify.title")}</h1>
            <p className="text-muted-foreground text-sm mt-1">{t("classify.subtitle")}</p>
          </div>
        </div>
        {file && (
          <Button variant="ghost" size="sm" onClick={clearAll}>
            <X className="mr-1 size-4" /> {t("common.back")}
          </Button>
        )}
      </div>

      {/* Models loading state */}
      {isLoading && (
        <Card>
          <CardContent className="pt-4 space-y-3">
            <Skeleton className="h-4 w-48" />
            <div className="flex flex-wrap gap-2">
              {[1,2,3].map(i => <Skeleton key={i} className="h-10 w-28 rounded-lg" />)}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Models error state */}
      {isError && (
        <Card className="border-destructive/50">
          <CardContent className="pt-4 text-center">
            <p className="text-sm text-destructive font-medium">{t("classify.models.load_error")}</p>
            <p className="text-xs text-muted-foreground mt-1">{error?.message}</p>
            <Button variant="outline" size="sm" className="mt-3" onClick={() => window.location.reload()}>
              <RefreshCw className="mr-1 size-3" /> {t("common.retry")}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Empty models state */}
      {!isLoading && !isError && models.length === 0 && (
        <Card className="border-dashed">
          <CardContent className="pt-8 pb-8 text-center">
            <HardDrive className="size-12 text-muted-foreground/30 mx-auto mb-3" />
            <p className="text-sm font-medium text-muted-foreground">{t("classify.models.empty")}</p>
            <p className="text-xs text-muted-foreground/70 mt-1">{t("classify.models.empty_hint")}</p>
          </CardContent>
        </Card>
      )}

      {/* Model selector — only when models exist */}
      {!isLoading && !isError && models.length > 0 && (
        <Card>
          <CardContent className="pt-4 space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                {t("classify.model_select.label")}
                <InfoTooltip text={t("classify.model_select.description")} />
              </p>

              {/* Cache actions */}
              <div className="flex items-center gap-2">
                {cache && (
                  <span className="text-xs text-muted-foreground">
                    {cache.cached_models}/{cache.total_models} cached
                  </span>
                )}
                <Button
                  variant="ghost" size="icon-sm"
                  onClick={() => downloadAllModels().then(() => cacheQuery.refetch())}
                  disabled={cache?.is_downloading}
                  title={t("classify.cache.download_all")}
                >
                  <Download className="size-3.5" />
                </Button>
                <Button
                  variant="ghost" size="icon-sm"
                  onClick={() => clearModelCache().then(() => cacheQuery.refetch())}
                  title={t("classify.cache.clear")}
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
            </div>

            {/* Download progress */}
            {cache?.is_downloading && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{t("classify.cache.downloading")}</span>
                  <span>{cache.download_progress.toFixed(0)}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                  <div className="h-full bg-primary rounded-full transition-all"
                    style={{ width: `${cache.download_progress}%` }} />
                </div>
              </div>
            )}

            {/* Model buttons grouped by experiment */}
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {Object.entries(grouped).map(([label, ms]) => (
                <div key={label}>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-1.5">
                    {label}
                  </h4>
                  <div className="flex flex-wrap gap-1.5">
                    {ms.map((m) => (
                      <button
                        key={m.id}
                        onClick={() => {
                          setSelectedModelId(m.id);
                          classify.reset();
                          explain.reset();
                          setHeatmapUrl(null);
                          setShowHeatmap(false);
                          // Trigger download if not cached
                          if (!m.is_cached) {
                            downloadModel(m.id).then(() => {
                              cacheQuery.refetch();
                              setSelectedModelId(m.id);
                            });
                          }
                        }}
                        className={cn(
                          "relative rounded-lg px-3 py-2 text-xs font-medium transition-colors min-h-[44px] cursor-pointer border",
                          selectedModelId === m.id
                            ? "bg-primary text-primary-foreground border-primary"
                            : "text-muted-foreground bg-muted/50 hover:bg-muted border-border/50",
                          !m.is_cached && "border-dashed",
                        )}
                      >
                        <span className="block text-left">{m.name}</span>
                        {m.mean_f1 != null && (
                          <span className="block text-[10px] opacity-70">
                            F1 {(m.mean_f1 * 100).toFixed(1)}%
                          </span>
                        )}
                        {/* Cache badge */}
                        <span className={cn(
                          "absolute top-1 right-1",
                          m.is_cached ? "text-green-500" : "text-amber-500",
                        )}>
                          {m.is_cached ? <HardDrive className="size-2.5" /> : <Cloud className="size-2.5" />}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Upload zone or classify panel */}
      {!file ? (
        <div
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-border py-12 sm:py-20 transition-colors hover:border-primary/50 hover:bg-muted/30"
        >
          <Upload className="mb-3 size-12 text-muted-foreground" />
          <p className="text-lg font-medium text-muted-foreground">{t("classify.upload.drop")}</p>
          <p className="mt-1 text-sm text-muted-foreground/70">
            {t("classify.upload.browse")} ({t("classify.upload.support")})
          </p>
          <input ref={inputRef} type="file" accept="image/*" onChange={onFileSelect} className="hidden" />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Image preview */}
          <Card>
            <CardHeader>
              <CardTitle>{showHeatmap ? t("classify.results.explainability") : t("classify.upload.title")}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative overflow-hidden rounded-lg bg-muted">
                <img src={showHeatmap && heatmapUrl ? heatmapUrl : previewUrl!}
                  alt={showHeatmap ? t("classify.results.heatmap_alt") : t("classify.upload.alt")}
                  className="mx-auto max-h-[500px] w-auto object-contain" />
              </div>
              {heatmapUrl && (
                <div className="mt-3 flex items-center gap-2">
                  <Button variant={showHeatmap ? "default" : "outline"} size="sm"
                    onClick={() => setShowHeatmap(true)}>
                    <Eye className="mr-1 size-4" /> {t("classify.results.heatmap_btn")}
                  </Button>
                  <Button variant={!showHeatmap ? "default" : "outline"} size="sm"
                    onClick={() => setShowHeatmap(false)}>
                    {t("classify.results.original")}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results panel */}
          <div className="space-y-4">
            {/* Classify + Explain buttons */}
            <Card>
              <CardContent className="space-y-3 pt-4">
                <div className="flex flex-wrap gap-3">
                  <Button onClick={handleClassify} disabled={isWorking}>
                    {classify.isPending ? <Loader2 className="mr-2 size-4 animate-spin" />
                      : <Sparkles className="mr-2 size-4" />}
                    {t("common.classify")}
                  </Button>
                  {classify.data && (
                    <Button variant="outline" onClick={handleExplain} disabled={isWorking}>
                      {explain.isPending ? <Loader2 className="mr-2 size-4 animate-spin" />
                        : <Eye className="mr-2 size-4" />}
                      {t("classify.actions.explain")}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Progress */}
            {isWorking && (
              <Card>
                <CardContent className="space-y-2 pt-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{activeLabel}</span>
                    <span className="font-mono text-xs text-muted-foreground">
                      {uploadDone ? t("classify.progress.please_wait")
                        : activeProgress > 0 ? `${activeProgress}%` : t("classify.progress.connecting")}
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    {uploadDone ? (
                      <div className="h-full w-full animate-pulse rounded-full bg-primary" />
                    ) : (
                      <div className="h-full rounded-full bg-primary transition-all duration-300"
                        style={{ width: `${Math.max(activeProgress, 5)}%` }} />
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Classification results */}
            {classify.data && (
              <Card>
                <CardHeader>
                  <CardTitle>{t("classify.results.predicted")} {classify.data.predicted_species}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {topPredictions.map((pred, i) => (
                    <div key={pred.species} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span className={cn("font-medium", i === 0 && "text-primary font-semibold")}>
                          {pred.species}
                        </span>
                        <span className="font-mono text-muted-foreground">
                          {(pred.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-muted">
                        <div className={cn("h-full rounded-full transition-all duration-500",
                          i === 0 ? "bg-primary" : "bg-primary/40")}
                          style={{ width: `${pred.confidence * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Explain results */}
            {explain.data && (
              <Card>
                <CardHeader>
                  <CardTitle>{t("classify.results.explainability")}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {t("classify.results.predicted")} <strong>{explain.data.predicted_species}</strong>
                    ({(explain.data.confidence * 100).toFixed(2)}%)
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {t("classify.results.heatmap_explanation")}
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Errors */}
            {classify.isError && (
              <Card className="border-destructive/50">
                <CardContent className="pt-4">
                  <p className="text-sm text-destructive">{t("common.error")}: {classify.error?.message}</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
