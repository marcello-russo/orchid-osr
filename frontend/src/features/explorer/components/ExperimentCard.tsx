// frontend/src/features/explorer/components/ExperimentCard.tsx

import { Link } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FlaskConical } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { ExperimentSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { formatPercent } from "@/lib/format";

interface ExperimentCardProps {
  experiment: ExperimentSummary;
}

export function ExperimentCard({ experiment }: ExperimentCardProps) {
  const { t } = useTranslation();
  const isKfold = experiment.type === "kfold";

  return (
    <Link to={`/experiments/${experiment.id}`} className="block cursor-pointer">
      <Card className="transition-colors duration-150 hover:border-primary/30">
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <CardTitle className="text-sm font-medium break-all">
              {experiment.name}
            </CardTitle>
            <Badge
              variant={isKfold ? "default" : "secondary"}
              className={cn(
                "text-xs shrink-0",
                isKfold && "bg-primary/10 text-primary hover:bg-primary/20"
              )}
            >
              {experiment.type}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {experiment.model_variant && (
              <span className="flex items-center gap-1">
                <FlaskConical className="h-3 w-3" />
                {experiment.model_variant}
              </span>
            )}
            {experiment.mean_f1 !== null && (
              <span>
                {`${t("experiment_detail.charts.f1_label")}:`} {formatPercent(experiment.mean_f1, 2)}
                {experiment.std_f1 != null && ` ± ${formatPercent(experiment.std_f1, 2)}`}
              </span>
            )}
            {experiment.test_accuracy !== null && (
              <span>
                {`${t("experiment_detail.charts.test_label")}:`} {formatPercent(experiment.test_accuracy, 2)}
              </span>
            )}
            {experiment.num_folds !== null && (
              <span>{experiment.num_folds} {t("experiment_detail.charts.folds_label")}</span>
            )}
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
