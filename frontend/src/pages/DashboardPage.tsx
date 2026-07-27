import { useState } from "react";
import { Link } from "react-router-dom";
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
  ScanSearch,
  Box,
  BookOpen,
  LayoutDashboard,
  ArrowRight,
} from "lucide-react";
import { useExperiments } from "@/features/explorer/hooks/useExperiments";
import { useSpecies } from "@/features/explorer/hooks/useSpecies";
import { KpiCard } from "@/features/explorer/components/KpiCard";
import { ExperimentCard } from "@/features/explorer/components/ExperimentCard";
import { Skeleton } from "@/components/ui/skeleton";

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
  const bestF1Model = cleanSplitExps.length > 0
    ? cleanSplitExps.reduce((best, e) => {
        const score = (e.mean_f1 ?? 0) - (e.std_f1 ?? 0);
        const bestScore = (best.mean_f1 ?? 0) - (best.std_f1 ?? 0);
        return score > bestScore ? e : best;
      })
    : null;
  const bestF1Val = bestF1Model?.mean_f1 ?? 0;
  const bestF1Std = bestF1Model?.std_f1 ?? 0;
  const ncdExps = (experiments ?? []).filter((e) => e.ncd_auroc != null);
  const bestNcdAuroc = ncdExps.length > 0
    ? ncdExps.reduce((max, e) => Math.max(max, e.ncd_auroc ?? 0), 0)
    : 0;
  const bestNcdNmi = ncdExps.length > 0
    ? ncdExps.reduce((max, e) => Math.max(max, e.ncd_clustering_nmi ?? 0), 0)
    : 0;

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 md:px-12 animate-in fade-in duration-700">
      <div className="flex items-center gap-4">
        <div className="size-10 md:size-12 rounded-2xl bg-primary/10 flex items-center justify-center shrink-0 text-primary">
          <LayoutDashboard className="size-6" />
        </div>
        <div>
          <h1 className="text-2xl md:text-4xl font-black tracking-tight">{t("dashboard.title")}</h1>
          <p className="text-muted-foreground text-sm mt-1">{t("dashboard.subtitle")}</p>
        </div>
      </div>

      {/* Banner */}
      <div>
        <img
          src="/assets/images/banner-osr.webp"
          alt="OrchID OSR"
          decoding="async"
          loading="eager"
          fetchPriority="high"
          className="w-full rounded-2xl"
        />
      </div>

      {/* Navigation Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <NavCard
          to="/experiments"
          icon={FlaskConical}
          title={t("common.experiments")}
          description={t("dashboard.nav.experiments")}
          color="blue"
        />
        <NavCard
          to="/classify"
          icon={ScanSearch}
          title={t("common.classify")}
          description={t("dashboard.nav.classify")}
          color="green"
        />
        <NavCard
          to="/dataset"
          icon={Database}
          title={t("common.dataset")}
          description={t("dashboard.nav.dataset")}
          color="amber"
        />
        <NavCard
          to="/embeddings"
          icon={Box}
          title={t("common.embeddings")}
          description={t("dashboard.nav.embeddings")}
          color="purple"
        />
        <NavCard
          to="/research"
          icon={BookOpen}
          title={t("common.about")}
          description={t("dashboard.nav.research")}
          color="primary"
          featured
        />
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
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
          title={t("dashboard.kpis.best_test_accuracy")}
          value={bestTestAcc > 0 ? `${(bestTestAcc * 100).toFixed(2)}%` : "—"}
          icon={TrendingUp}
          delay={100}
        />
        <KpiCard
          title={t("dashboard.kpis.best_f1")}
          value={bestF1Val > 0 ? `${(bestF1Val * 100).toFixed(2)}% ± ${(bestF1Std * 100).toFixed(2)}%` : "—"}
          icon={BarChart3}
          delay={150}
        />
        <KpiCard
          title={t("dashboard.kpis.classes")}
          value={speciesData ? String(speciesData.total_species) : "—"}
          subtitle={t("dashboard.kpis.in_dataset")}
          icon={Flower2}
          delay={200}
        />
        <KpiCard
          title={t("dashboard.kpis.total_images")}
          value={speciesData ? String(speciesData.total_images) : "—"}
          subtitle={t("dashboard.kpis.combined_dataset")}
          icon={Database}
          delay={250}
        />
      </div>

      {/* NCD KPI Row (Open-World) */}
      {ncdExps.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-4">{t("dashboard.kpis.open_world")}</h2>
          <div className="grid grid-cols-2 gap-3">
            <KpiCard
              title={t("dashboard.kpis.best_ncd_auroc")}
              value={bestNcdAuroc > 0 ? `${(bestNcdAuroc * 100).toFixed(2)}%` : "—"}
              icon={ScanSearch}
              delay={300}
            />
            <KpiCard
              title={t("dashboard.kpis.best_ncd_nmi")}
              value={bestNcdNmi > 0 ? bestNcdNmi.toFixed(4) : "—"}
              icon={BarChart3}
              delay={350}
            />
          </div>
        </div>
      )}

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
            {t("dashboard.species_distribution")}
          </button>
          {datasetOpen && (
            <div className="rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-4 py-3 text-left font-medium text-muted-foreground">{t("dashboard.table.species")}</th>
                    <th className="px-4 py-3 text-right font-medium text-muted-foreground">{t("dashboard.table.count")}</th>
                    <th className="px-4 py-3 text-right font-medium text-muted-foreground">{t("dashboard.table.percentage")}</th>
                  </tr>
                </thead>
                <tbody>
                  {speciesData.species.map((s) => (
                    <tr key={s.name} className="border-b border-border last:border-0">
                      <td className="px-4 py-2.5 font-medium">{s.name}</td>
                      <td className="px-4 py-2.5 text-right font-mono">{s.image_count}</td>
                      <td className="px-4 py-2.5 text-right font-mono">
                        {((s.image_count / speciesData.total_images) * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
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

const colorMap: Record<string, { bg: string; text: string; hoverBorder: string }> = {
  blue:   { bg: "bg-blue-500/10",    text: "text-blue-500",    hoverBorder: "hover:border-blue-500/30" },
  green:  { bg: "bg-green-500/10",   text: "text-green-500",   hoverBorder: "hover:border-green-500/30" },
  amber:  { bg: "bg-amber-500/10",   text: "text-amber-500",   hoverBorder: "hover:border-amber-500/30" },
  purple: { bg: "bg-purple-500/10",  text: "text-purple-500",  hoverBorder: "hover:border-purple-500/30" },
  primary:{ bg: "bg-primary/10",     text: "text-primary",     hoverBorder: "hover:border-primary/30" },
};

function NavCard({
  to,
  icon: Icon,
  title,
  description,
  color,
  featured = false,
}: {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  color: string;
  featured?: boolean;
}) {
  const c = colorMap[color] ?? colorMap.primary;

  return (
    <Link
      to={to}
      className={`group relative flex flex-col gap-3 rounded-xl border border-border/50 bg-card p-5 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md cursor-pointer ${c.hoverBorder} ${
        featured ? "sm:col-span-2 lg:col-span-1 ring-1 ring-primary/20 bg-primary/[0.03]" : ""
      }`}
    >
      <div className="flex items-start justify-between">
        <div className={`size-10 rounded-lg ${c.bg} flex items-center justify-center ${c.text}`}>
          <Icon className="size-5" />
        </div>
        <ArrowRight className="size-4 text-muted-foreground/0 group-hover:text-muted-foreground transition-colors mt-1" />
      </div>
      <div>
        <h3 className={`font-semibold ${featured ? "text-primary" : ""}`}>{title}</h3>
        <p className="mt-1 text-sm text-muted-foreground leading-relaxed">{description}</p>
      </div>
    </Link>
  );
}
