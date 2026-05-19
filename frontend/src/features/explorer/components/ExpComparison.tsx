import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { fetchExpComparison } from "@/lib/api";
import type { ExperimentComparisonResponse } from "@/lib/types";

type Tab = "exp6" | "exp7" | "confronto";

function formatPct(v: number | null): string {
  if (v === null) return "—";
  return (v * 100).toFixed(2) + "%";
}

function formatF1(f1: number | null, std: number | null): string {
  if (f1 === null) return "—";
  const base = (f1 * 100).toFixed(2);
  return std !== null ? `${base} ± ${(std * 100).toFixed(2)}` : base;
}

export function ExpComparison() {
  const { t } = useTranslation();
  const [tab, setTab] = useState<Tab>("exp6");

  const { data, isLoading } = useQuery<ExperimentComparisonResponse>({
    queryKey: ["comparison", "series"],
    queryFn: fetchExpComparison,
    staleTime: 5 * 60 * 1000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="size-6 rounded-full border-2 border-primary border-t-transparent animate-spin" />
      </div>
    );
  }

  const exp7Empty = !data?.exp7.length;

  return (
    <div className="space-y-6">
      {/* Tabs */}
      <div className="flex gap-1 border-b border-border/50">
        {[
          { id: "exp6" as Tab, label: "Exp 6 (CE)" },
          { id: "exp7" as Tab, label: "Exp 7 (SupCon)" },
          { id: "confronto" as Tab, label: "Confronto" },
        ].map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`px-4 py-2.5 text-xs font-bold uppercase tracking-wider transition-colors border-b-2 -mb-px ${
              tab === id
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      {tab === "exp6" && <SerieTable models={data?.exp6 ?? []} label="Exp 6 (CE)" />}
      {tab === "exp7" && (
        exp7Empty ? (
          <div className="text-center py-16 space-y-3">
            <p className="text-muted-foreground text-sm">{t("about.experiments.no_data")}</p>
            <p className="text-xs text-muted-foreground/50 max-w-md mx-auto">{t("about.experiments.hint")}</p>
          </div>
        ) : (
          <SerieTable models={data!.exp7} label="Exp 7 (SupCon)" />
        )
      )}
      {tab === "confronto" && (
        exp7Empty ? (
          <div className="text-center py-16 space-y-3">
            <p className="text-muted-foreground text-sm">{t("about.experiments.no_data")}</p>
            <p className="text-xs text-muted-foreground/50 max-w-md mx-auto">{t("about.experiments.hint")}</p>
          </div>
        ) : (
          <DeltaTable delta={data!.delta} />
        )
      )}
    </div>
  );
}

function SerieTable({ models, label }: { models: import("@/lib/types").ModelComparisonEntry[]; label: string }) {
  if (!models.length) {
    return (
      <div className="text-center py-16 text-sm text-muted-foreground">
        Nessun modello disponibile per {label}.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-2xl border border-border/50 bg-background/50">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border/50 text-xs uppercase tracking-widest text-muted-foreground font-bold">
            <th className="text-left py-4 px-5">#</th>
            <th className="text-left py-4 px-3">{label}</th>
            <th className="text-center py-4 px-3">F1</th>
            <th className="text-center py-4 px-3">Accuracy</th>
            <th className="text-center py-4 pr-5 pl-3">Best Fold</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/30">
          {models.map((m) => (
            <tr key={m.model_variant} className={m.rank === 1 ? "bg-amber-500/5" : ""}>
              <td className="py-3.5 px-5 font-bold tabular-nums">
                {m.rank === 1 ? "🥇" : m.rank === 2 ? "🥈" : m.rank === 3 ? "🥉" : `#${m.rank}`}
              </td>
              <td className="py-3.5 px-3 font-bold">{m.name}</td>
              <td className="text-center py-3.5 px-3 tabular-nums">{formatF1(m.mean_f1, m.std_f1)}</td>
              <td className="text-center py-3.5 px-3 tabular-nums">{formatPct(m.mean_accuracy)}</td>
              <td className="text-center py-3.5 pr-5 pl-3 tabular-nums text-muted-foreground text-xs">
                {m.best_fold !== null ? `Fold ${m.best_fold}` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DeltaTable({ delta }: { delta: import("@/lib/types").SeriesModelEntry[] }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-border/50 bg-background/50">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border/50 text-xs uppercase tracking-widest text-muted-foreground font-bold">
            <th className="text-left py-4 px-5">Modello</th>
            <th className="text-center py-4 px-3">Exp 6 (CE)</th>
            <th className="text-center py-4 px-3">Exp 7 (SupCon)</th>
            <th className="text-center py-4 pr-5 pl-3">Δ</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/30">
          {delta.map((d) => {
            const isBetter = d.delta !== null && d.delta > 0;
            const isWorse = d.delta !== null && d.delta < 0;
            return (
              <tr key={d.variant}>
                <td className="py-3.5 px-5 font-bold">{d.display_name}</td>
                <td className="text-center py-3.5 px-3 tabular-nums">{formatPct(d.f1_ce)}</td>
                <td className="text-center py-3.5 px-3 tabular-nums">
                  {d.f1_supcon !== null ? formatPct(d.f1_supcon) : "—"}
                </td>
                <td className={`text-center py-3.5 pr-5 pl-3 tabular-nums font-bold ${
                  isBetter ? "text-green-600" : isWorse ? "text-red-500" : "text-muted-foreground"
                }`}>
                  {d.delta !== null ? `${isBetter ? "+" : ""}${d.delta.toFixed(1)}` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
