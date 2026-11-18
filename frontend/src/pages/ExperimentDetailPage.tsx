import { useState } from "react";
import { useParams, Link, useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ArrowLeft, GitCompare } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useExperiment } from "@/features/explorer/hooks/useExperiment";
import { FoldSelector } from "@/features/explorer/components/FoldSelector";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import { ExperimentConfigCard } from "@/features/explorer/components/ExperimentConfigCard";
import { getTooltips } from "@/features/explorer/tooltipText";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useCompareExperiments } from "@/features/explorer/hooks/useCompareExperiments";
import { CompareModeView } from "@/features/explorer/components/CompareModeView";
import { ExperimentSelectTable } from "@/features/explorer/components/ExperimentSelectTable";
import { ChartCarousel } from "@/features/explorer/detail/ChartCarousel";
import { BaselineResults } from "@/features/explorer/detail/BaselineResults";
import { NcdDetail } from "@/features/explorer/detail/NcdDetail";
import { LogsPanel } from "@/features/explorer/detail/LogsPanel";

export function ExperimentDetailPage() {
  const { t } = useTranslation();
  const TOOLTIPS = getTooltips(t);
  const { id } = useParams<{ id: string }>();
  const expNum = id ? parseInt(id.match(/exp(\d+)/)?.[1] ?? "", 10) : NaN;
  const { data: experiment, isLoading, error } = useExperiment(id ?? "");
  const [searchParams, setSearchParams] = useSearchParams();
  const compareRaw = searchParams.get("compare") ?? "";
  const compareIds = compareRaw ? compareRaw.split(",").filter((x) => x && x !== id) : [];
  const isCompareMode = compareIds.length >= 1;
  const { experiments: compareExperiments, isLoading: compareLoading } = useCompareExperiments(
    isCompareMode ? compareIds : []
  );
  const [selectedFold, setSelectedFold] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState("charts");

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-96" />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Skeleton className="h-[350px] rounded-lg" />
          <Skeleton className="h-[350px] rounded-lg" />
        </div>
      </div>
    );
  }

  if (error || !experiment) {
    return (
      <div className="py-20 text-center text-muted-foreground">
        <p>{t("common.error")}</p>
        <Link to="/experiments" className="mt-2 inline-block text-primary hover:underline">
          {t("experiment_detail.back_to_experiments")}
        </Link>
      </div>
    );
  }

  const { config, folds, results, summary } = experiment;
  const hasFolds = folds.length > 0;
  const hasResults = results !== null;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div>
        <Link
          to="/experiments"
          className="mb-2 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-primary transition-colors duration-150"
        >
          <ArrowLeft className="h-4 w-4" />
          {t("common.experiments")}
        </Link>
        <div className="flex flex-wrap items-center gap-2">
          {isCompareMode ? (
            <>
              <span className="text-lg sm:text-2xl font-bold">{t("experiment.comparing")}</span>
              {[experiment, ...compareExperiments].filter(Boolean).map((exp, i) => {
                const COLORS = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6"];
                return (
                  <Badge key={exp.id} variant="outline" className="font-mono text-xs"
                    style={{ borderColor: COLORS[i % 4], color: COLORS[i % 4] }}>
                    {exp.id}
                  </Badge>
                );
              })}
            </>
          ) : (
            <>
              <h1 className="text-lg sm:text-2xl font-bold text-primary break-all">{id}</h1>
              <Badge variant="secondary" className="text-xs">
                {summary.type}
              </Badge>
              {summary.model_variant && (
                <Badge variant="outline" className="font-mono text-xs">
                  {summary.model_variant}
                </Badge>
              )}
            </>
          )}
        </div>
        <div className="mt-2 flex items-center gap-2">
          {!isCompareMode ? (
            <Dialog>
              <DialogTrigger render={<Button variant="outline" size="sm" />}>
                <GitCompare className="mr-1.5 size-4" /> {t("experiment.compare")}
              </DialogTrigger>
              <DialogContent className="sm:max-w-2xl md:max-w-3xl lg:max-w-4xl">
                <DialogHeader>
                  <DialogTitle>{t("experiment.select_compare")}</DialogTitle>
                </DialogHeader>
                <ExperimentSelectTable
                  currentId={id!}
                  onCompare={(selected) => setSearchParams({ compare: selected.join(",") })}
                />
              </DialogContent>
            </Dialog>
          ) : (
            <Button variant="ghost" size="sm" onClick={() => setSearchParams({})}>
              {t("experiment.exit_compare")}
            </Button>
          )}
        </div>

        {/* Config summary — only in single-experiment mode */}
        {!isCompareMode && (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted-foreground">

          {summary.test_accuracy !== null && (
            <span className="font-medium text-primary">
              {t("experiment_detail.metrics.accuracy")}: {(summary.test_accuracy * 100).toFixed(2)}%
              <InfoTooltip text={TOOLTIPS.testAccuracy} />
            </span>
          )}
          {summary.test_f1 !== null && (
            <span className="font-medium text-primary">
              {t("experiment_detail.metrics.f1")}: {(summary.test_f1 * 100).toFixed(2)}%
              <InfoTooltip text={TOOLTIPS.f1macro} />
            </span>
          )}
        </div>
        )}
      </div>

      {/* Training recipe card — replaced by CompareModeView in compare mode */}
      {isCompareMode && compareLoading ? (
        <Skeleton className="h-40 rounded-lg" />
      ) : isCompareMode ? (
        <CompareModeView
          experiments={[experiment, ...compareExperiments].filter(Boolean)}
        />
      ) : (
        <ExperimentConfigCard config={config} />
      )}

      {/* Fold selector — hidden in compare mode */}
      {hasFolds && !isCompareMode && (
        <FoldSelector
          folds={folds}
          selectedFold={selectedFold}
          onSelect={setSelectedFold}
        />
      )}

      {/* Tab bar — hidden in compare mode */}
      {!isCompareMode && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setActiveTab("charts")}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
              activeTab === "charts" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
            }`}
          >
            {t("experiment.charts")}
          </button>
          {(expNum === 9) && (
            <button
              onClick={() => setActiveTab("ncd")}
              className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
                activeTab === "ncd" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
              }`}
            >
              {t("experiment_detail.ncd_tab")}
            </button>
          )}
          <button
            onClick={() => setActiveTab("logs")}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
              activeTab === "logs" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
            }`}
          >
            {t("experiment.logs")}
          </button>
        </div>
      )}

      {/* Content — hidden in compare mode */}
      {!isCompareMode && activeTab === "charts" && (hasFolds ? (
        <>
        <ChartCarousel
          folds={folds}
          selectedFold={selectedFold}
          meanF1={summary.mean_f1}
          stdF1={summary.std_f1}
          confusionMatrix={
            hasResults && results.test_metrics
              ? { matrix: results.test_metrics.confusion_matrix, labels: results.test_metrics.confusion_matrix_labels }
              : null
          }
          experimentId={id}
        />
        </>
      ) : hasResults && results.test_metrics ? (
        <BaselineResults metrics={results.test_metrics} />
      ) : (
        <div className="py-10 text-center text-muted-foreground">
          {t("experiment_detail.no_results")}
        </div>
      ))}

      {!isCompareMode && activeTab === "ncd" && expNum === 9 && (
        <NcdDetail modelVariant={summary.model_variant ?? config.model_variant ?? ""} exp={expNum} />
      )}

      {!isCompareMode && activeTab === "logs" && id && (
        <LogsPanel experimentId={id} />
      )}
    </div>
  );
}
