import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, ReferenceLine,
} from "recharts";

const CHART_COLORS = ["#3B82F6", "#8B5CF6", "#F59E0B", "#EF4444"];
const CLUSTER_PALETTE = ["#3B82F6", "#8B5CF6", "#22C55E", "#EF4444"];

interface AblationData {
  name: string;
  auroc: number;
  fill: string;
}

interface ClusterData {
  algorithm: string;
  acc: number;
  nmi: number;
  ari: number;
  f_measure: number;
}

interface PerClassAUROC {
  left_out: string;
  auroc: number;
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
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis dataKey="name" tick={{ fill: "#888", fontSize: 12 }} />
          <YAxis domain={[0, 0.6]} tick={{ fill: "#888", fontSize: 11 }} />
          <ReferenceLine y={0.5} stroke="#666" strokeDasharray="5 5" label={{ value: "Random (0.50)", fill: "#666", fontSize: 11 }} />
          <Bar dataKey="auroc" radius={[4, 4, 0, 0]}>
            {data.map((_, i) => <Cell key={i} fill={CHART_COLORS[i]} />)}
          </Bar>
          <Tooltip
            contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }}
            formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "AUROC"]}
          />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

export function ClusteringComparisonChart({ clustering }: { clustering: Record<string, any> }) {
  const { t } = useTranslation();
  const algos = ["kmeans_best", "kmeans_true_k", "spectral", "dbscan"];
  const labels = ["K-Means (best k)", "K-Means (k=true)", "Spectral", "DBSCAN"];
  const metrics = [
    { key: "acc", label: "ACC" },
    { key: "nmi", label: "NMI" },
    { key: "ari", label: "ARI" },
    { key: "f_measure", label: "F" },
  ];

  const chartData = useMemo(() =>
    metrics.map((m) => ({
      name: m.label,
      ...Object.fromEntries(
        algos.map((a, i) => [labels[i], clustering?.[a]?.[m.key] ?? 0])
      ),
    })),
  [clustering, metrics]);

  if (!chartData.length) return null;

  return (
    <section>
      <h3 className="text-lg font-bold mb-4">{t("experiment_detail.ncd_clustering")}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis dataKey="name" tick={{ fill: "#888", fontSize: 12 }} />
          <YAxis domain={[0, 1]} tick={{ fill: "#888", fontSize: 11 }} />
          {labels.map((l, i) => (
            <Bar key={l} dataKey={l} fill={CLUSTER_PALETTE[i]} radius={[2, 2, 0, 0]} />
          ))}
          <Tooltip
            contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }}
            formatter={(value: number) => [(value * 100).toFixed(1) + "%"]}
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
      name: p.left_out.replace("O. ", "").replace("O. sphegodes_", "s."),
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
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis dataKey="name" tick={{ fill: "#888", fontSize: 10 }} angle={-20} textAnchor="end" />
          <YAxis domain={[0, 0.6]} tick={{ fill: "#888", fontSize: 11 }} />
          <ReferenceLine y={0.5} stroke="#666" strokeDasharray="5 5" label={{ value: t("experiment_detail.ncd_random_chance"), fill: "#666", fontSize: 11 }} />
          <Bar dataKey="auroc" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.isNovel ? "#EF4444" : "#3B82F6"} />
            ))}
          </Bar>
          <Tooltip
            contentStyle={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8 }}
            formatter={(value: number) => [`${(value * 100).toFixed(1)}%`]}
          />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}
