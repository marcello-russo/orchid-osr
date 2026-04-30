// frontend/src/pages/DashboardPage.tsx

import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Activity,
  FlaskConical,
  TrendingUp,
  BarChart3,
  Database,
  Flower2,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { useExperiments } from "@/features/explorer/hooks/useExperiments";
import { useSpecies } from "@/features/explorer/hooks/useSpecies";
import { KpiCard } from "@/features/explorer/components/KpiCard";
import { ExperimentCard } from "@/features/explorer/components/ExperimentCard";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export function DashboardPage() {
  const { t } = useTranslation();
  const { data: experiments, isLoading: expLoading } = useExperiments();
  const { data: speciesData, isLoading: speciesLoading } = useSpecies();
  const [datasetOpen, setDatasetOpen] = useState(true);

  const cleanSplitExps = (experiments ?? []).filter(
    (e) => e.type === "kfold" && e.id.startsWith("exp6")
  );
  const completedCount = (experiments ?? []).filter(
    (e) => e.status === "completed"
  ).length;
  const bestTestAcc = cleanSplitExps.reduce(
    (max, e) => Math.max(max, e.test_accuracy ?? 0),
    0
  );
  const bestMeanF1 = cleanSplitExps.reduce(
    (max, e) => Math.max(max, e.mean_f1 ?? 0),
    0
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <h1 className="text-2xl font-bold text-primary">{t("dashboard.title")}</h1>

      {/* KPI Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <KpiCard
          title={t("experiments.title")}
          value={String(experiments?.length ?? 0)}
          icon={FlaskConical}
          delay={0}
        />
        <KpiCard
          title={t("dashboard.kpis.completed_models")}
          value={String(completedCount)}
          icon={Activity}
          delay={50}
        />
        <KpiCard
          title="Best Test Accuracy"
          value={bestTestAcc > 0 ? `${(bestTestAcc * 100).toFixed(2)}%` : "—"}
          icon={TrendingUp}
          delay={100}
        />
        <KpiCard
          title={t("dashboard.kpis.best_f1")}
          value={bestMeanF1 > 0 ? `${(bestMeanF1 * 100).toFixed(2)}%` : "—"}
          icon={BarChart3}
          delay={150}
        />
        <KpiCard
          title={t("dashboard.kpis.classes")}
          value={speciesData ? String(speciesData.total_species) : "—"}
          subtitle="in dataset"
          icon={Flower2}
          delay={200}
        />
        <KpiCard
          title={t("dashboard.kpis.total_images")}
          value={speciesData ? String(speciesData.total_images) : "—"}
          subtitle="combined dataset"
          icon={Database}
          delay={250}
        />
      </div>

      {/* Dataset Overview */}
      {speciesData && speciesData.species.length > 0 && (
        <div>
          <button
            onClick={() => setDatasetOpen((o) => !o)}
            className="mb-4 flex items-center gap-1 text-lg font-medium transition-colors hover:text-primary cursor-pointer"
          >
            {datasetOpen ? (
              <ChevronDown className="size-5" />
            ) : (
              <ChevronRight className="size-5" />
            )}
            🌿 {t("dashboard.species_distribution")}
          </button>
          {datasetOpen && (
            <div className="rounded-lg border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("dashboard.table.species")}</TableHead>
                    <TableHead className="text-right">{t("dashboard.table.count")}</TableHead>
                    <TableHead className="text-right">{t("dashboard.table.percentage")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {speciesData.species.map((s) => (
                    <TableRow key={s.name}>
                      <TableCell className="font-medium">{s.name}</TableCell>
                      <TableCell className="text-right font-mono">
                        {s.image_count}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {((s.image_count / speciesData.total_images) * 100).toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </div>
      )}
      {speciesLoading && (
        <div className="space-y-2">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-32 w-full rounded-lg" />
        </div>
      )}

      {/* Recent Experiments */}
      <div>
        <h2 className="mb-4 text-lg font-semibold">{t("dashboard.recent_experiments")}</h2>
        {expLoading ? (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-24 w-full rounded-lg" />
            ))}
          </div>
        ) : (experiments ?? []).length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <FlaskConical className="mb-3 size-10 opacity-30" />
            <p className="text-sm">{t("common.error")}</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            {(experiments ?? [])
              .slice()
              .reverse()
              .slice(0, 6)
              .map((exp) => (
                <ExperimentCard key={exp.id} experiment={exp} />
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
