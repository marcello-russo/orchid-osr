// frontend/src/pages/ExperimentDetailPage.tsx

import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useExperiment } from "@/features/explorer/hooks/useExperiment";
import { FoldSelector } from "@/features/explorer/components/FoldSelector";
import { LossCurvesChart } from "@/features/explorer/components/LossCurvesChart";
import { MetricCurvesChart } from "@/features/explorer/components/MetricCurvesChart";
import { FoldSummaryChart } from "@/features/explorer/components/FoldSummaryChart";
import { ConfusionMatrixChart } from "@/features/explorer/components/ConfusionMatrixChart";
import { OverfittingProxyChart } from "@/features/explorer/components/OverfittingProxyChart";

export function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data: experiment, isLoading, error } = useExperiment(id ?? "");
  const [selectedFold, setSelectedFold] = useState<number | null>(null);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-96" />
        <div className="grid grid-cols-2 gap-4">
          <Skeleton className="h-[350px] rounded-lg" />
          <Skeleton className="h-[350px] rounded-lg" />
        </div>
      </div>
    );
  }

  if (error || !experiment) {
    return (
      <div className="py-20 text-center text-muted-foreground">
        <p>Experiment not found.</p>
        <Link to="/experiments" className="mt-2 inline-block text-primary hover:underline">
          Back to experiments
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
          Experiments
        </Link>
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold text-primary">{id}</h1>
          <Badge variant="secondary" className="text-xs">
            {summary.type}
          </Badge>
          {summary.model_variant && (
            <Badge variant="outline" className="font-mono text-xs">
              {summary.model_variant}
            </Badge>
          )}
        </div>

        {/* Config summary */}
        <div className="mt-2 flex flex-wrap gap-4 text-sm text-muted-foreground">
          {config.num_epochs && <span>Epochs: {config.num_epochs}</span>}
          {config.batch_size && <span>Batch: {config.batch_size}</span>}
          {config.learning_rate && <span>LR: {config.learning_rate}</span>}
          {config.patience !== undefined && <span>Patience: {config.patience}</span>}
          {summary.test_accuracy !== null && (
            <span className="font-medium text-primary">
              Test Accuracy: {(summary.test_accuracy * 100).toFixed(1)}%
            </span>
          )}
          {summary.test_f1 !== null && (
            <span className="font-medium text-accent">
              Test F1: {(summary.test_f1 * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {/* Fold selector — only for kfold experiments */}
      {hasFolds && (
        <FoldSelector
          folds={folds}
          selectedFold={selectedFold}
          onSelect={setSelectedFold}
        />
      )}

      {/* Charts grid */}
      {hasFolds ? (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <LossCurvesChart folds={folds} selectedFold={selectedFold} />
          <MetricCurvesChart folds={folds} selectedFold={selectedFold} />
          <FoldSummaryChart
            folds={folds}
            meanF1={summary.mean_f1}
            stdF1={summary.std_f1}
          />
          <OverfittingProxyChart folds={folds} selectedFold={selectedFold} />
        </div>
      ) : (
        <div className="py-10 text-center text-muted-foreground">
          Baseline experiment — no fold data available.
        </div>
      )}

      {/* Confusion matrix — only if test metrics exist */}
      {hasResults && results.test_metrics && (
        <ConfusionMatrixChart
          matrix={results.test_metrics.confusion_matrix}
          labels={results.test_metrics.confusion_matrix_labels}
        />
      )}
    </div>
  );
}
