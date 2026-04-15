// frontend/src/pages/DashboardPage.tsx

import { Activity, FlaskConical, TrendingUp, BarChart3 } from "lucide-react";
import { useExperiments } from "@/features/explorer/hooks/useExperiments";
import { KpiCard } from "@/features/explorer/components/KpiCard";
import { ExperimentCard } from "@/features/explorer/components/ExperimentCard";
import { Skeleton } from "@/components/ui/skeleton";

export function DashboardPage() {
  const { data: experiments, isLoading } = useExperiments();

  const kfoldExps = (experiments ?? []).filter((e) => e.type === "kfold");
  const completedCount = (experiments ?? []).filter(
    (e) => e.status === "completed"
  ).length;
  const bestTestAcc = kfoldExps.reduce(
    (max, e) => Math.max(max, e.test_accuracy ?? 0),
    0
  );
  const bestMeanF1 = kfoldExps.reduce(
    (max, e) => Math.max(max, e.mean_f1 ?? 0),
    0
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <h1 className="text-2xl font-semibold text-primary">🌸 Dashboard</h1>

      {/* KPI Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          title="Total Experiments"
          value={String(experiments?.length ?? 0)}
          icon={FlaskConical}
          delay={0}
        />
        <KpiCard
          title="Completed"
          value={String(completedCount)}
          icon={Activity}
          delay={50}
        />
        <KpiCard
          title="Best Test Accuracy"
          value={bestTestAcc > 0 ? `${(bestTestAcc * 100).toFixed(1)}%` : "—"}
          icon={TrendingUp}
          delay={100}
        />
        <KpiCard
          title="Best Mean F1"
          value={bestMeanF1 > 0 ? `${(bestMeanF1 * 100).toFixed(1)}%` : "—"}
          icon={BarChart3}
          delay={150}
        />
      </div>

      {/* Recent Experiments */}
      <div>
        <h2 className="mb-3 text-lg font-medium">Recent Experiments</h2>
        {isLoading ? (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-24 w-full rounded-lg" />
            ))}
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
