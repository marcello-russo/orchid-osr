import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, ReferenceLine,
} from "recharts";

import type { NcdClusteringResult } from "@/lib/types";

const CHART_COLORS = ["#3B82F6", "#8B5CF6", "#F59E0B", "#EF4444"];
const CLUSTER_PALETTE = ["#3B82F6", "#8B5CF6", "#22C55E", "#EF4444"];
const CLUSTER_ALGOS = ["kmeans_best", "kmeans_true_k", "spectral", "dbscan"];
const CLUSTER_LABELS = ["K-Means (best k)", "K-Means (k=true)", "Spectral", "DBSCAN"];
const CHART_METRICS = [
  { key: "acc" as const, label: "ACC" },
  { key: "nmi" as const, label: "NMI" },
  { key: "ari" as const, label: "ARI" },
  { key: "f_measure" as const, label: "F" },
];

interface AblationData {
  name: string;
  auroc: number;
  fill: string;
}

interface PerClassAUROC {
  left_out: string;
  auroc: number;
}

function shortLabel(species: string): string {
  return species.replace(/^O\.\s*/, "").replace(/^sphegodes_/, "s.");
}

export function AblationChart({ ablation }: { ablation: Record<string, number> }) {
  const { t } = useTranslation();
  const data: AblationData[] = useMemo(() => {
    const methods = ["mahalanobis", "knn", "energy", "ensemble"];
    return methods.map((m, i) => ({
      name: m.charAt(0).toUpperCase() + m.slice(1),
      auroc: ablation?.[m] ?? 0,
      fill: CHART_COLORS[i],
    }));
  }, [ablation]);

  if (!data.length) return null;

  return (
    <section>
      <h3 className="text-lg font-bold mb-4">{t("experiment_detail.ncd_ablation")}</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="name" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
          <YAxis domain={[0, 0.6]} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
          <ReferenceLine y={0.5} stroke="hsl(var(--muted-foreground))" strokeDasharray="5 5" label={{ value: "Random (0.50)", fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
          <Bar dataKey="auroc" radius={[4, 4, 0, 0]}>
            {data.map((_, i) => <Cell key={i} fill={CHART_COLORS[i]} />)}
          </Bar>
          <Tooltip
            contentStyle={{ background: "hsl(var(--background))", border: "1px solid hsl(var(--border))", borderRadius: 8 }}
            formatter={(value) => [`${(Number(value ?? 0) * 100).toFixed(1)}%`, "AUROC"]}
          />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

export function ClusteringComparisonChart({ clustering }: { clustering: Record<string, NcdClusteringResult> }) {
  const { t } = useTranslation();

  const chartData = useMemo(() =>
    CHART_METRICS.map((m) => ({
      name: m.label,
      ...Object.fromEntries(
        CLUSTER_ALGOS.map((a, i) => [CLUSTER_LABELS[i], clustering?.[a]?.[m.key] ?? 0])
      ),
    })),
  [clustering]);

  if (!chartData.length) return null;

  return (
    <section>
      <h3 className="text-lg font-bold mb-4">{t("experiment_detail.ncd_clustering")}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="name" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
          <YAxis domain={[0, 1]} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
          {CLUSTER_LABELS.map((l, i) => (
            <Bar key={l} dataKey={l} fill={CLUSTER_PALETTE[i]} radius={[2, 2, 0, 0]} />
          ))}
          <Tooltip
            contentStyle={{ background: "hsl(var(--background))", border: "1px solid hsl(var(--border))", borderRadius: 8 }}
            formatter={(value) => [`${(Number(value ?? 0) * 100).toFixed(1)}%`]}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

export function PerClassAurocChart({
  perClass,
  napoliAuroc,
}: {
  perClass: PerClassAUROC[];
  napoliAuroc?: number;
}) {
  const { t } = useTranslation();
  const data = useMemo(() => {
    const items = perClass.map((p) => ({
      name: shortLabel(p.left_out),
      auroc: p.auroc,
      isNovel: false,
    }));
    if (napoliAuroc !== undefined) {
      items.push({ name: "s. Napoli*", auroc: napoliAuroc, isNovel: true });
    }
    return items;
  }, [perClass, napoliAuroc]);

  if (!data.length) return null;

  return (
    <section>
      <h3 className="text-lg font-bold mb-4">
        {t("experiment_detail.ncd_osr")} — AUROC per class
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="name" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }} angle={-20} textAnchor="end" />
          <YAxis domain={[0, 0.6]} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
          <ReferenceLine y={0.5} stroke="hsl(var(--muted-foreground))" strokeDasharray="5 5" label={{ value: t("experiment_detail.ncd_random_chance"), fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
          <Bar dataKey="auroc" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.isNovel ? "#EF4444" : "#3B82F6"} />
            ))}
          </Bar>
          <Tooltip
            contentStyle={{ background: "hsl(var(--background))", border: "1px solid hsl(var(--border))", borderRadius: 8 }}
            formatter={(value) => [`${(Number(value ?? 0) * 100).toFixed(1)}%`]}
          />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}
