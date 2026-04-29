// frontend/src/pages/ClassifyPage.tsx

import { useState, useRef, useCallback } from "react";
import { Upload, Loader2, Sparkles, Eye, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useClassify } from "@/features/classify/hooks/useClassify";
import { useExplain } from "@/features/classify/hooks/useExplain";
import { useModels } from "@/features/classify/hooks/useModels";
import { cn } from "@/lib/utils";
import { InfoTooltip } from "@/components/ui/InfoTooltip";

export function ClassifyPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const classify = useClassify();
  const explain = useExplain();
  const { data: modelsData } = useModels();
  const models = modelsData?.models ?? [];

  const isWorking = classify.isPending || explain.isPending;
  const activeProgress = classify.isPending
    ? classify.progress
    : explain.isPending
      ? explain.progress
      : 0;
  const uploadDone = isWorking && activeProgress >= 100;
  const activeLabel = uploadDone
    ? "Processing on server..."
    : classify.isPending
      ? "Uploading..."
      : explain.isPending
        ? "Uploading..."
        : "";

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

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const onFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleClassify = () => {
    if (!file) return;
    classify.mutate({ file, modelId: selectedModelId ?? undefined });
  };

  const handleExplain = () => {
    if (!file) return;
    explain.mutate(
      { file, modelId: selectedModelId ?? undefined },
      {
        onSuccess: (data) => {
          setHeatmapUrl(`data:image/png;base64,${data.heatmap_base64}`);
          setShowHeatmap(true);
        },
      }
    );
  };

  const clearAll = () => {
    setFile(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setHeatmapUrl(null);
    setShowHeatmap(false);
    classify.reset();
    explain.reset();
    if (inputRef.current) inputRef.current.value = "";
  };

  const topPredictions = classify.data?.top_predictions ?? [];

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-primary">
          🔍 Classify
        </h1>
        {file && (
          <Button variant="ghost" size="sm" onClick={clearAll}>
            <X className="mr-1 size-4" /> Clear
          </Button>
        )}
      </div>

      {/* Upload zone */}
      {!file ? (
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onClick={() => inputRef.current?.click()}
          className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-border py-12 sm:py-20 transition-colors hover:border-primary/50 hover:bg-muted/30"
        >
          <Upload className="mb-3 size-12 text-muted-foreground" />
          <p className="text-lg font-medium text-muted-foreground">
            Drag & drop an orchid photo here
          </p>
          <p className="mt-1 text-sm text-muted-foreground/70">
            or click to browse (JPG, PNG)
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={onFileSelect}
            className="hidden"
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Image preview + heatmap */}
          <Card>
            <CardHeader>
              <CardTitle>
                {showHeatmap ? "Explainability Heatmap" : "Uploaded Image"}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative overflow-hidden rounded-lg bg-muted">
                <img
                  src={showHeatmap && heatmapUrl ? heatmapUrl : previewUrl!}
                  alt={showHeatmap ? "Heatmap overlay" : "Uploaded orchid"}
                  className="mx-auto max-h-[500px] w-auto object-contain"
                />
              </div>
              {heatmapUrl && (
                <div className="mt-3 flex items-center gap-2">
                  <Button
                    variant={showHeatmap ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowHeatmap(true)}
                  >
                    <Eye className="mr-1 size-4" /> Heatmap
                  </Button>
                  <Button
                    variant={!showHeatmap ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowHeatmap(false)}
                  >
                    Original
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results panel */}
          <div className="space-y-4">
            {/* Model selector + actions */}
            <Card>
              <CardContent className="space-y-3 pt-4">
                {models.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs font-medium text-muted-foreground flex items-center">
                      Model
                      <InfoTooltip text="Only models from Experiment 6 (clean split, 6 classes) are available for classification." />
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {models.map((m) => (
                        <button
                          key={m.id}
                          onClick={() => {
                            setSelectedModelId(m.id);
                            classify.reset();
                            explain.reset();
                            setHeatmapUrl(null);
                            setShowHeatmap(false);
                          }}
                          className={cn(
                            "rounded-lg px-3 py-2 text-xs font-medium transition-colors min-h-[44px]",
                            selectedModelId === m.id
                              ? "bg-primary text-primary-foreground"
                              : "text-muted-foreground bg-muted/50 hover:bg-muted"
                          )}
                        >
                          {m.model_variant}
                          {m.mean_f1 !== null && (
                            <span className="ml-1 opacity-70">
                              F1 {(m.mean_f1 * 100).toFixed(1)}%
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <div className="flex flex-wrap gap-3">
                  <Button
                    onClick={handleClassify}
                    disabled={isWorking}
                  >
                    {classify.isPending ? (
                      <Loader2 className="mr-2 size-4 animate-spin" />
                    ) : (
                      <Sparkles className="mr-2 size-4" />
                    )}
                    Classify
                  </Button>
                  {classify.data && (
                    <Button
                      variant="outline"
                      onClick={handleExplain}
                      disabled={isWorking}
                    >
                      {explain.isPending ? (
                        <Loader2 className="mr-2 size-4 animate-spin" />
                      ) : (
                        <Eye className="mr-2 size-4" />
                      )}
                      Explain
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Progress bar */}
            {isWorking && (
              <Card>
                <CardContent className="space-y-2 pt-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{activeLabel}</span>
                    <span className="font-mono text-xs text-muted-foreground">
                      {uploadDone ? "please wait" : activeProgress > 0 ? `${activeProgress}%` : "Connecting..."}
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    {uploadDone ? (
                      <div className="h-full w-full animate-pulse rounded-full bg-primary" />
                    ) : (
                      <div
                        className="h-full rounded-full bg-primary transition-all duration-300"
                        style={{ width: `${Math.max(activeProgress, 5)}%` }}
                      />
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Classification results */}
            {classify.data && (
              <Card>
                <CardHeader>
                  <CardTitle>
                    🌸 {classify.data.predicted_species}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {topPredictions.map((pred, i) => (
                    <div key={pred.species} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span
                          className={cn(
                            "font-medium",
                            i === 0 && "text-primary font-semibold"
                          )}
                        >
                          {pred.species}
                        </span>
                        <span className="font-mono text-muted-foreground">
                          {(pred.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-muted">
                        <div
                          className={cn(
                            "h-full rounded-full transition-all duration-500",
                            i === 0
                              ? "bg-primary"
                              : "bg-primary/40"
                          )}
                          style={{ width: `${pred.confidence * 100}%` }}
                        />
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
                  <CardTitle>🔬 Explainability</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Predicted: <strong>{explain.data.predicted_species}</strong>{" "}
                    ({(explain.data.confidence * 100).toFixed(2)}%)
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    The heatmap highlights which regions of the image most
                    influenced the model's prediction.{" "}
                    <span className="text-green-600">Green</span> = positive
                    contribution, <span className="text-red-600">Red</span> =
                    negative.
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Error states */}
            {classify.isError && (
              <Card className="border-destructive/50">
                <CardContent className="pt-4">
                  <p className="text-sm text-destructive">
                    Classification failed: {classify.error?.message}
                  </p>
                </CardContent>
              </Card>
            )}
            {explain.isError && (
              <Card className="border-destructive/50">
                <CardContent className="pt-4">
                  <p className="text-sm text-destructive">
                    Explainability failed: {explain.error?.message}
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
