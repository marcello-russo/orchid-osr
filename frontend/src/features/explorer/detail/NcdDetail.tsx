// frontend/src/features/explorer/detail/NcdDetail.tsx

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { fetchNcdResults } from "@/lib/api";
import { AblationChart, ClusteringComparisonChart, PerClassAurocChart } from "@/features/explorer/components/NcdCharts";
import { Scatter2D } from "@/features/embeddings/Scatter2D";
import { Scatter3D } from "@/features/embeddings/Scatter3D";
import type { NcdClusteringResult } from "@/lib/types";


const SPECIES_PALETTE: Record<string, string> = {
  "O. exaltata": "#3B82F6", "O. garganica": "#D97706", "O. incubacea": "#10B981",
  "O. majellensis": "#8B5CF6", "O. sphegodes": "#EF4444", "O. sphegodes_Palena": "#EC4899",
  "novel": "#14B8A6", "unknown": "#666666",
};

export function NcdDetail({ modelVariant, exp }: { modelVariant: string; exp?: number }) {
  const { t } = useTranslation();
  const [tsneMode, setTsneMode] = useState<"2d" | "3d">("2d");
  const { data, isLoading, error } = useQuery({
    queryKey: ["ncd", modelVariant, exp ?? 8],
    queryFn: () => fetchNcdResults(modelVariant, exp ?? 8),
    staleTime: 5 * 60 * 1000,
    enabled: modelVariant.length > 0,
  });

  if (isLoading) return <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>;
  if (error || !data) return <p className="text-sm text-muted-foreground text-center py-8">{t("experiment_detail.ncd_no_data")}</p>;

  const detectors = data.detectors;
  const openset = data.openset;
  const clustering = data.clustering ?? {};
  const meanAuroc = detectors?.Ensemble?.auroc ?? openset?.mean_auroc ?? 0;
  const novelAuroc = detectors?.Ensemble?.auroc ?? openset?.novel_class?.auroc;
  const ablation = openset?.ablation;
  const perClass = openset?.per_class_metrics ?? [];
  const tsne2d = data.tsne_2d as [number, number][] | undefined;
  const tsne3d = data.tsne_3d as [number, number, number][] | undefined;
  const labels = data.novel_class_label as number[] | undefined;

  // Nome della novel class: usato da class_names (novel_class è una classe
  // nota per le exp non-NCD, mentre per exp 9 è la classe nascosta).
  const novelClassName = data.novel_class ?? "novel";

  // Compute best clustering ACC
  const bestClustering = Object.values(clustering).reduce<NcdClusteringResult | null>(
    (best, c) => (c.acc && c.acc > (best?.acc ?? 0) ? c : best), null
  );
  const silhouette = clustering.kmeans_best?.silhouette ?? null;

  // Build scatter points for t-SNE — map known classes to species names
  const classNames: string[] = data.class_names ?? [];
  const scatterPoints = tsne2d?.map((p, i) => ({
    x: p[0], y: p[1],
    species: labels?.[i] !== undefined && labels[i] >= 0 && labels[i] < classNames.length
      ? classNames[labels[i]] : novelClassName,
    idx: i,
  })) ?? [];

  const speciesSet = [...new Set(scatterPoints.map((p) => p.species))];
  const speciesColors = speciesSet.length > 0
    ? Object.fromEntries(speciesSet.map((s) => [s, SPECIES_PALETTE[s] ?? "#666"]))
    : {};

  return (
    <div className="space-y-8 p-4">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-xl border border-border/40 bg-background p-4 text-center">
          <div className="text-xs text-muted-foreground uppercase font-bold mb-1">{t("experiment_detail.ncd_auroc")}</div>
          <div className={`text-2xl font-black ${meanAuroc < 0.5 ? "text-red-500" : "text-green-500"}`}>
            {(meanAuroc * 100).toFixed(1)}%
          </div>
          <div className="flex flex-wrap justify-center gap-1.5 mt-1.5">
            {["Ensemble", "Mahalanobis", "KNN", "Energy"].map((d) => {
              const val = detectors?.[d]?.auroc;
              return val != null ? (
                <span key={d} className="text-[10px] px-1.5 py-0.5 rounded bg-muted font-mono">
                  {d[0]}: {(val * 100).toFixed(1)}%
                </span>
              ) : null;
            })}
          </div>
        </div>
        {novelAuroc !== undefined && (
          <div className="rounded-xl border-2 border-red-500/30 bg-red-500/5 p-4 text-center">
            <div className="text-xs text-muted-foreground uppercase font-bold mb-1">
              {t("experiment_detail.ncd_novel_auroc")}
              <span className="ml-1.5 inline-block bg-red-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded">
                {t("experiment_detail.ncd_novel_label")}
              </span>
            </div>
            <div className="text-2xl font-black text-red-500">{(novelAuroc * 100).toFixed(1)}%</div>
            <div className="text-[10px] text-muted-foreground mt-0.5">{t("experiment.vs_random")}</div>
          </div>
        )}
        <div className="rounded-xl border border-border/40 bg-background p-4 text-center">
          <div className="text-xs text-muted-foreground uppercase font-bold mb-1">NMI</div>
          <div className="text-2xl font-black text-purple-500">
            {bestClustering?.nmi ? bestClustering.nmi.toFixed(3) : "—"}
          </div>
          <div className="text-[10px] text-muted-foreground mt-0.5">clustering</div>
        </div>
        <div className="rounded-xl border border-border/40 bg-background p-4 text-center">
          <div className="text-xs text-muted-foreground uppercase font-bold mb-1">{t("experiment_detail.ncd_best_acc")}</div>
          <div className="text-2xl font-black text-green-500">
            {bestClustering?.acc ? `${(bestClustering.acc * 100).toFixed(1)}%` : "—"}
          </div>
          <div className="text-[10px] text-muted-foreground mt-0.5">{bestClustering?.k ? `k=${bestClustering.k}` : ""}</div>
        </div>
        <div className="rounded-xl border border-border/40 bg-background p-4 text-center">
          <div className="text-xs text-muted-foreground uppercase font-bold mb-1">{t("experiment_detail.ncd_silhouette")}</div>
          <div className={`text-2xl font-black ${(silhouette ?? 0) > 0.1 ? "text-yellow-500" : "text-muted-foreground"}`}>
            {silhouette !== null ? silhouette.toFixed(3) : "—"}
          </div>
          <div className="text-[10px] text-muted-foreground mt-0.5">
            {clustering.kmeans_best?.k ? `k=${clustering.kmeans_best.k}` : ""}
          </div>
        </div>
      </div>

      {/* Ablation chart */}
      {ablation && <AblationChart ablation={ablation} />}

      {/* t-SNE viewer */}
      {scatterPoints.length > 0 && (
        <section>
          <h3 className="text-lg font-bold mb-4">
            {t("experiment.tsne")}
            <div className="inline-flex ml-3 gap-1">
              <button onClick={() => setTsneMode("2d")}
                className={`px-2 py-0.5 text-xs rounded font-bold ${tsneMode === "2d" ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
                {t("experiment_detail.ncd_tsne_toggle_2d")}
              </button>
              <button onClick={() => setTsneMode("3d")}
                className={`px-2 py-0.5 text-xs rounded font-bold ${tsneMode === "3d" ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
                {t("experiment_detail.ncd_tsne_toggle_3d")}
              </button>
            </div>
          </h3>
          <div className="rounded-xl border border-border/50 bg-background/50 p-4">
            {tsneMode === "2d" ? (
              <Scatter2D points={scatterPoints} speciesColors={speciesColors} />
            ) : tsne3d ? (
              <Scatter3D
                points={scatterPoints.map((p, i) => ({ ...p, z: tsne3d[i]?.[2] ?? 0 }))}
                species={[...new Set(scatterPoints.map((p) => p.species))]}
                speciesColors={speciesColors}
              />
            ) : (
              <p className="text-sm text-muted-foreground">{t("experiment.tsne_unavailable")}</p>
            )}
          </div>
        </section>
      )}

      {/* Clustering comparison chart */}
      {Object.keys(clustering).length > 0 && <ClusteringComparisonChart clustering={clustering} />}

      {/* Clustering table */}
      <section>
        <h3 className="text-lg font-bold mb-4">{t("experiment_detail.ncd_clustering")}</h3>
        <div className="overflow-x-auto rounded-xl border border-border/50">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50 text-xs uppercase text-muted-foreground font-bold">
                <th className="text-left py-3 px-4">{t("experiment_detail.ncd_algorithm")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_acc")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_nmi")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_ari")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_ami")}</th>
                <th className="text-center py-3 px-2">{t("experiment_detail.ncd_clustering_f")}</th>
                <th className="text-center py-3 px-2">{t("experiment.k_label")}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30">
              {Object.entries(clustering).map(([key, val]) => (
                <tr key={key}>
                  <td className="py-2 px-4 font-bold">{key}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.acc?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.nmi?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.ari?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.ami?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.f_measure?.toFixed(4) ?? "—"}</td>
                  <td className="text-center py-2 px-2 font-mono">{val.k ?? val.n_clusters ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* OSR section */}
      <section>
        <h3 className="text-lg font-bold mb-4">{t("experiment_detail.ncd_osr")}</h3>

        {/* AUROC per class chart */}
        <PerClassAurocChart perClass={perClass} novelAuroc={novelAuroc} />

        {/* Per-class table */}
        {perClass.length > 0 && (
          <div className="overflow-x-auto rounded-xl border border-border/50 mt-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/50 text-xs uppercase text-muted-foreground font-bold">
                  <th className="text-left py-3 px-4">{t("experiment_detail.ncd_class")}</th>
                  <th className="text-center py-3 px-2">AUROC</th>
                  <th className="text-center py-3 px-2">{t("experiment_detail.ncd_known_acc")}</th>
                  <th className="text-center py-3 px-2">{t("experiment_detail.ncd_unknown_recall")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {perClass.map((pc) => (
                  <tr key={pc.left_out}
                    className={pc.left_out === novelClassName || pc.left_out.toLowerCase().includes("garganica") ? "bg-red-500/10 font-bold" : ""}>
                    <td className="py-2 px-4 font-bold">{pc.left_out}</td>
                    <td className="text-center py-2 px-2 font-mono">{(pc.auroc * 100).toFixed(1)}%</td>
                    <td className="text-center py-2 px-2 font-mono">{(pc.known_acc * 100).toFixed(1)}%</td>
                    <td className="text-center py-2 px-2 font-mono">{(pc.unknown_recall * 100).toFixed(1)}%</td>
                  </tr>
                ))}
                {novelAuroc !== undefined && (
                  <tr className="bg-red-500/10 font-bold">
                    <td className="py-2 px-4 font-bold">
                      {novelClassName}
                      <span className="ml-1.5 inline-block bg-red-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded">
                        {t("experiment_detail.ncd_novel_label")}
                      </span>
                    </td>
                    <td className="text-center py-2 px-2 font-mono text-red-500">{(novelAuroc * 100).toFixed(1)}%</td>
                    <td className="text-center py-2 px-2 font-mono">—</td>
                    <td className="text-center py-2 px-2 font-mono">
                      {openset?.novel_class?.unknown_recall ? `${(openset.novel_class.unknown_recall * 100).toFixed(1)}%` : "—"}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
