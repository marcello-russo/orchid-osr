import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  Loader2, Sparkles, X, ScanSearch,
  Download, HardDrive, RefreshCw, ChevronDown,
  CheckCircle, AlertCircle, Flower2, Image as ImageIcon,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Card, CardContent,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useClassify } from "@/features/classify/hooks/useClassify";
import { useModels } from "@/features/classify/hooks/useModels";
import {
  fetchCacheStatus, downloadAllModels, downloadModel,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import type { ModelOption } from "@/lib/types";

/* ── Tag display ── */

const TAG_LABELS: Record<string, { label: string; hint: string; variant: "default" | "secondary" | "outline" }> = {
  ncd: { label: "NCD", hint: "Can detect novel/unseen Ophrys species", variant: "secondary" },
  best_f1: { label: "★ Best F1", hint: "Highest classification accuracy across all models", variant: "default" },
};

function TagBadge({ tag }: { tag: string }) {
  const meta = TAG_LABELS[tag];
  if (!meta) return null;
  return (
    <Tooltip>
      <TooltipTrigger>
        <Badge variant={meta.variant} className="text-[10px] px-1.5 py-0 cursor-help">
          {meta.label}
        </Badge>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-48 text-balance text-xs">
        {meta.hint}
      </TooltipContent>
    </Tooltip>
  );
}

/* ── Model Select Dropdown ── */

function ModelSelect({
  models, selected, onChange, cache, onCacheRefresh,
}: {
  models: ModelOption[];
  selected: string | null;
  onChange: (id: string) => void;
  cache: { cached_models: number; total_models: number; is_downloading: boolean; download_progress: number } | undefined;
  onCacheRefresh: () => void;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const sel = models.find((m) => m.id === selected);

  const grouped: Record<string, ModelOption[]> = {};
  models.forEach((m) => {
    const key = m.exp_label || `Exp ${m.exp_num ?? "?"}`;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(m);
  });

  return (
    <TooltipProvider delay={200}>
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full rounded-xl border border-border bg-card px-4 py-3 text-sm font-medium hover:border-primary/40 transition-colors min-h-[48px] cursor-pointer"
      >
        <span className="truncate">
          {sel ? (
            <span className="flex items-center gap-2">
              <Flower2 className="size-4 text-primary shrink-0" />
              <span>{sel.name}</span>
              <Badge variant="outline" className="text-[10px] font-mono px-1.5 py-0">
                F1 {(sel.mean_f1! * 100).toFixed(1)}%
              </Badge>
              {sel.tags?.map((tag) => <TagBadge key={tag} tag={tag} />)}
              {!sel.is_cached && (
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                  {t("classify.cache.not_cached")}
                </Badge>
              )}
            </span>
          ) : (
            <span className="text-muted-foreground">{t("classify.model_select.label")}</span>
          )}
        </span>
        <ChevronDown className={cn("size-4 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute left-0 right-0 top-full mt-1 z-50 rounded-xl border border-border bg-card shadow-xl max-h-72 overflow-y-auto">
            {/* Cache summary */}
            {cache && (
              <div className="sticky top-0 bg-card border-b border-border px-3 py-2 flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  {cache.cached_models}/{cache.total_models} {t("classify.cache.cached")}
                </span>
                <div className="flex gap-2">
                  {cache.is_downloading ? (
                    <span className="text-primary">{cache.download_progress.toFixed(0)}%</span>
                  ) : (
                    <button className="hover:text-foreground cursor-pointer flex items-center gap-1" onClick={() => { downloadAllModels().then(onCacheRefresh); }}>
                      <Download className="size-3" /> {t("classify.cache.download_all")}
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Groups */}
            {Object.entries(grouped).map(([label, ms]) => (
              <div key={label}>
                <div className="px-3 pt-2 pb-1 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
                  {label}
                </div>
                {ms.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => { onChange(m.id); setOpen(false); }}
                    className={cn(
                      "w-full flex items-center gap-3 px-3 py-2.5 text-sm text-left transition-colors hover:bg-muted/50 cursor-pointer",
                      selected === m.id && "bg-primary/5",
                    )}
                  >
                    <div className="size-7 rounded-md bg-primary/10 flex items-center justify-center shrink-0">
                      {m.is_cached
                        ? <HardDrive className="size-3.5 text-primary" />
                        : <Download className="size-3.5 text-muted-foreground" />
                      }
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate">{m.name}</div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span className="text-[11px] text-muted-foreground">
                          F1 {(m.mean_f1! * 100).toFixed(1)}%
                        </span>
                        {m.tags?.map((tag) => <TagBadge key={tag} tag={tag} />)}
                      </div>
                    </div>
                    {selected === m.id && <CheckCircle className="size-4 text-primary shrink-0" />}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
    </TooltipProvider>
  );
}

/* ── Empty Drop State ── */

function DropZone({ onClick, onDrop, onDragOver, onChange, inputRef, t }: {
  onClick: () => void;
  onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  t: (key: string) => string;
}) {
  return (
    <div
      onDrop={onDrop}
      onDragOver={onDragOver}
      onClick={onClick}
      className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border bg-card/50 py-16 sm:py-24 transition-all hover:border-primary/50 hover:bg-primary/[0.02] group"
    >
      <div className="size-16 rounded-2xl bg-primary/5 flex items-center justify-center mb-5 group-hover:bg-primary/10 transition-colors">
        <ImageIcon className="size-7 text-primary/60" />
      </div>
      <p className="text-lg font-semibold text-foreground">{t("classify.upload.drop")}</p>
      <p className="text-sm text-muted-foreground mt-1">
        {t("classify.upload.browse")}
      </p>
      <p className="text-xs text-muted-foreground/60 mt-4 font-mono">
        {t("classify.upload_hint")}
      </p>
      <input ref={inputRef} type="file" accept="image/*" onChange={onChange} className="hidden" />
    </div>
  );
}

/* ── Main Page ── */

export function ClassifyPage() {
  const { t } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const classify = useClassify();
  const { data: modelsData, isLoading, isError, error } = useModels();
  const models = useMemo(() => modelsData?.models ?? [], [modelsData?.models]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  useEffect(() => {
    if (!selectedModelId && models.length > 0) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSelectedModelId(models[0].id);
    }
  }, [models, selectedModelId]);

  const cacheQuery = useQuery({
    queryKey: ["cache-status"],
    queryFn: fetchCacheStatus,
    refetchInterval: (q) => q.state.data?.is_downloading ? 2000 : false,
  });

  const isWorking = classify.isPending;
  const cache = cacheQuery.data;

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    setPreviewUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(f); });
    classify.reset();
  }, [classify]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const dropProps = {
    onDrop, onDragOver: (e: React.DragEvent) => e.preventDefault(),
    onClick: () => inputRef.current?.click(),
    onChange: onFileSelect, inputRef, t,
  };

  const handleClassify = () => {
    if (!file) return;
    if (!selectedModelId && models.length > 0) {
      setSelectedModelId(models[0].id);
    }
    classify.mutate({ file, modelId: selectedModelId ?? undefined });
  };

  const clearAll = () => {
    setFile(null);
    setPreviewUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; });
    classify.reset();
    if (inputRef.current) inputRef.current.value = "";
  };

  const topPredictions = classify.data?.top_predictions ?? [];

  return (
    <div className="max-w-5xl mx-auto px-4 md:px-8 py-8 md:py-12 animate-in fade-in duration-500 space-y-6">
      {/* ── Header ── */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary shrink-0">
            <ScanSearch className="size-5 md:size-6" />
          </div>
          <div>
            <h1 className="text-2xl md:text-4xl font-black tracking-tight">{t("classify.title")}</h1>
            <p className="text-muted-foreground text-sm mt-1">{t("classify.subtitle")}</p>
          </div>
        </div>
        {file && (
          <Button variant="ghost" size="sm" onClick={clearAll}>
            <X className="mr-1.5 size-4" /> {t("common.back")}
          </Button>
        )}
      </div>

      {/* ── Loading ── */}
      {isLoading && (
        <Card>
          <CardContent className="pt-6 space-y-4">
            <Skeleton className="h-5 w-48" />
            <Skeleton className="h-12 w-full rounded-xl" />
          </CardContent>
        </Card>
      )}

      {/* ── Error ── */}
      {isError && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="pt-6 text-center space-y-3">
            <AlertCircle className="size-10 text-destructive mx-auto" />
            <p className="font-medium text-destructive">{t("classify.models.load_error")}</p>
            <p className="text-xs text-muted-foreground">{error?.message}</p>
            <Button variant="outline" size="sm" onClick={() => window.location.reload()}>
              <RefreshCw className="mr-1.5 size-3" /> {t("common.retry")}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* ── Empty ── */}
      {!isLoading && !isError && models.length === 0 && (
        <Card className="border-dashed">
          <CardContent className="pt-12 pb-12 text-center space-y-3">
            <HardDrive className="size-12 text-muted-foreground/20 mx-auto" />
            <p className="font-medium">{t("classify.models.empty")}</p>
            <p className="text-xs text-muted-foreground">{t("classify.models.empty_hint")}</p>
            <Button variant="outline" size="sm" className="mt-2" onClick={() => {
              downloadAllModels().then(() => window.location.reload());
            }}>
              <Download className="mr-1.5 size-3" /> {t("classify.cache.download_all")}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* ── Model Select + Cache ── */}
      {!isLoading && !isError && models.length > 0 && (
        <div className="space-y-3">
          <ModelSelect
            models={models}
            selected={selectedModelId}
            onChange={(id) => {
              setSelectedModelId(id);
              classify.reset();
              const m = models.find((x) => x.id === id);
              if (m && !m.is_cached) {
                downloadModel(m.id).then(() => cacheQuery.refetch());
              }
            }}
            cache={cache}
            onCacheRefresh={() => cacheQuery.refetch()}
          />

          {/* Download progress bar */}
          {cache?.is_downloading && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{t("classify.cache.downloading")}</span>
                <span>{cache.download_progress.toFixed(0)}%</span>
              </div>
              <div className="h-1 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-primary rounded-full transition-all" style={{ width: `${cache.download_progress}%` }} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Main Content ── */}
      {models.length > 0 && !file && (
        <DropZone {...dropProps} />
      )}

      {/* ── Results Grid ── */}
      {models.length > 0 && file && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Image */}
          <Card className="lg:col-span-3 overflow-hidden">
            <div className="relative bg-muted/30">
              <img
                src={previewUrl!}
                alt={t("classify.upload.alt")}
                className="w-full h-full object-contain max-h-[480px]"
              />

            </div>
          </Card>

          {/* Results */}
          <div className="lg:col-span-2 space-y-4">
            {/* Actions */}
            <div className="flex gap-3">
              <Button onClick={handleClassify} disabled={isWorking || classify.isSuccess} className="flex-1">
                {classify.isPending ? <Loader2 className="mr-2 size-4 animate-spin" /> : <Sparkles className="mr-2 size-4" />}
                {t("common.classify")}
              </Button>
            </div>

            {/* Upload progress */}
            {isWorking && !classify.data && (
              <Card>
                <CardContent className="pt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">
                      {classify.isPending ? t("classify.progress.uploading") : t("common.loading")}
                    </span>
                    <span className="font-mono text-xs text-muted-foreground">
                      {classify.progress > 0 ? `${classify.progress}%` : t("classify.progress.connecting")}
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                    <div className="h-full bg-primary rounded-full transition-all" style={{ width: `${Math.max(classify.progress, 3)}%` }} />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Classification Results */}
            {classify.data && (
              <div className="space-y-4">
                {classify.data.novelty?.is_novel ? (
                  /* Novel — show warning only, hide predictions */
                  <div className="rounded-lg border border-amber-200 bg-amber-50 p-6 dark:border-amber-800 dark:bg-amber-950">
                    <div className="flex items-start gap-4">
                      <span className="text-3xl">⚠️</span>
                      <div>
                        <p className="text-lg font-bold text-amber-800 dark:text-amber-200">
                          Possible New Species
                        </p>
                        <p className="text-sm text-amber-700 dark:text-amber-300 mt-2">
                          This flower does not match any known Ophrys species with sufficient confidence.
                          It may represent an undocumented taxon or a variant not present in the training set.
                        </p>
                        <p className="text-xs text-amber-600 dark:text-amber-400 mt-3">
                          Closest match: {classify.data.predicted_species} ({(classify.data.confidence * 100).toFixed(1)}%)
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Known — show predictions */
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <span className="text-lg">✅</span>
                      <span className="text-2xl font-bold">{classify.data.predicted_species}</span>
                      <span className="text-lg text-muted-foreground">
                        {(classify.data.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="space-y-2">
                      {topPredictions.map((p, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <span className="w-36 text-sm truncate shrink-0">{p.species}</span>
                          <div className="flex-1 h-6 rounded bg-muted overflow-hidden">
                            <div className="h-full rounded bg-primary/80 transition-all"
                              style={{ width: `${p.confidence * 100}%` }} />
                          </div>
                          <span className="w-14 text-xs text-right text-muted-foreground shrink-0">
                            {(p.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error */}
            {classify.isError && (
              <Card className="border-destructive/30">
                <CardContent className="pt-4 space-y-2">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="size-4 text-destructive mt-0.5 shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-destructive">{t("common.error")}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">{classify.error?.message}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
