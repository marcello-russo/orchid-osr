import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useComparison } from "@/features/explorer/hooks/useComparison";
import { useTranslation } from "react-i18next";
import { useChartColors } from "@/features/explorer/hooks/useChartColors";
import { fetchComparisonFold } from "@/lib/api";
import { foldToEpochPoints } from "@/features/explorer/types";
import type { FoldData } from "@/lib/types";

function MicrosoftLogo() {
  return <img src="/microsoft-logo.webp" alt="Microsoft" className="size-full object-scale-down p-1" />;
}

function MetaLogo() {
  return <img src="/meta-logo.webp" alt="Meta" className="h-full max-w-full object-scale-down p-1" />;
}

const FAMILY: Record<string, { label: string; logo: React.ReactNode }> = {
  resnet18:        { label: "ResNet / Microsoft", logo: <MicrosoftLogo /> },
  resnet50:        { label: "ResNet / Microsoft", logo: <MicrosoftLogo /> },
  convnext_tiny:   { label: "ConvNeXt / Meta",   logo: <MetaLogo /> },
  convnext_small:  { label: "ConvNeXt / Meta",   logo: <MetaLogo /> },
  dinov2_small:    { label: "DINOv2 / Meta",     logo: <MetaLogo /> },
  dinov2_base:     { label: "DINOv2 / Meta",     logo: <MetaLogo /> },
};

const MEDAL = ["🥇", "🥈", "🥉"];

function formatPct(v: number | null, decimals = 2): string {
  if (v === null || v === undefined) return "—";
  return (v * 100).toFixed(decimals) + "%";
}

function formatF1(f1: number | null, std: number | null): string {
  if (f1 === null) return "—";
  const base = (f1 * 100).toFixed(2);
  if (std === null) return base;
  return `${base} ± ${(std * 100).toFixed(2)}`;
}

export function ResultsTable() {
  const { t } = useTranslation();
  const { data, isLoading, error } = useComparison();
  const colors = useChartColors();
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggle = useCallback((variant: string) => {
    setExpanded((prev) => (prev === variant ? null : variant));
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="size-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-sm text-muted-foreground text-center py-12">
        {t("about.results.unavailable")}
      </div>
    );
  }

  const { models } = data;
  if (models.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-12">
        {t("about.results.no_data")}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="overflow-x-auto rounded-2xl border border-border/50 bg-background/50">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border/50 text-xs uppercase tracking-widest text-muted-foreground font-bold">
              <th className="text-left py-4 px-5 w-12">{t("about.results.rank")}</th>
              <th className="text-left py-4 px-3">{t("about.results.model")}</th>
              <th className="text-center py-4 px-3">{t("about.results.f1")}</th>
              <th className="text-center py-4 px-3">{t("about.results.accuracy")}</th>
              <th className="text-center py-4 px-3">{t("about.results.test_acc")}</th>
              <th className="text-center py-4 pr-2 pl-3">{t("about.results.best_fold")}</th>
              <th className="text-right py-4 pr-5 w-10" />
            </tr>
          </thead>
          <tbody className="divide-y divide-border/30">
            {models.map((m) => (
              <ModelRow
                key={m.model_variant}
                model={m}
                isExpanded={expanded === m.model_variant}
                onToggle={toggle}
                colors={colors}
              />
            ))}
          </tbody>
        </table>
      </div>

      <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
        {(() => {
          const best = models.find((m) => m.rank === 1);
          if (!best || best.mean_f1 === null) return null;
          return <CommentLine text={t("about.results.comment_winner", { model: best.name, f1: formatPct(best.mean_f1) })} />;
        })()}

        {(() => {
          const stable = models.reduce<(typeof models)[number] | null>((b, m) => {
            if (m.std_f1 === null) return b;
            if (!b || m.std_f1 < b.std_f1!) return m;
            return b;
          }, null);
          if (stable && stable.std_f1 !== null && stable.std_f1 < 0.01) {
            return <CommentLine text={t("about.results.comment_stability", { model: stable.name, std: (stable.std_f1 * 100).toFixed(2) })} />;
          }
          return null;
        })()}

        {(() => {
          const f1s = models.filter((m) => m.mean_f1 !== null).map((m) => m.mean_f1!);
          if (f1s.length >= 2) {
            const gap = (f1s[0] - f1s[f1s.length - 1]) * 100;
            if (gap > 1) {
              return <CommentLine text={t("about.results.comment_gap", {
                gap: gap.toFixed(1),
                top: models.find((m) => m.rank === 1)?.name ?? "",
                bottom: models.filter((m) => m.mean_f1 !== null).pop()?.name ?? "",
              })} />;
            }
          }
          return null;
        })()}
      </div>
    </div>
  );
}

function ModelRow({
  model,
  isExpanded,
  onToggle,
  colors,
}: {
  model: import("@/lib/types").ModelComparisonEntry;
  isExpanded: boolean;
  onToggle: (variant: string) => void;
  colors: string[];
}) {
  const fam = FAMILY[model.model_variant] || { label: "?", logo: <span>?</span> };
  const colorIdx = Math.min(Object.keys(FAMILY).indexOf(model.model_variant), colors.length - 1);
  const color = colors[Math.max(0, colorIdx)];

  return (
    <>
      <tr
        onClick={() => onToggle(model.model_variant)}
        className={`cursor-pointer transition-colors hover:bg-muted/30 ${model.rank === 1 ? "bg-amber-500/5" : ""}`}
      >
        <td className="py-4 px-5">
          {model.rank ? (
            <span className="text-lg font-black tabular-nums">
              {model.rank <= 3 ? MEDAL[model.rank - 1] : `#${model.rank}`}
            </span>
          ) : (
            <span className="text-muted-foreground text-xs">—</span>
          )}
        </td>
        <td className="py-4 px-3">
          <div className="flex items-center gap-3">
            <span
              className="size-8 rounded-lg flex items-center justify-center shrink-0 bg-background/80 ring-1 ring-border/60"
            >
              {fam.logo}
            </span>
            <div>
              <div className="font-bold text-foreground">{model.name}</div>
              <div className="text-[11px] text-muted-foreground tracking-wider uppercase">
                {fam.label}
              </div>
            </div>
          </div>
        </td>
                  <td className="text-center py-4 px-3">
                    <span className="font-bold tabular-nums">{formatF1(model.mean_f1, model.std_f1)}</span>
                  </td>
                  <td className="text-center py-4 px-3 tabular-nums">
                    {formatPct(model.mean_accuracy)}
                  </td>
                  <td className="text-center py-4 px-3 tabular-nums text-xs text-muted-foreground">
                    {formatPct(model.test_accuracy)}
                  </td>
                  <td className="text-center py-4 pr-2 pl-3 tabular-nums text-muted-foreground text-xs">
                    {model.best_fold !== null ? `Fold ${model.best_fold}` : "—"}
                  </td>
                  <td className="text-right py-4 pr-5 text-muted-foreground">
          {isExpanded ? <ChevronDown className="size-4 inline" /> : <ChevronRight className="size-4 inline" />}
        </td>
      </tr>
      {isExpanded && (
        <tr key={`${model.model_variant}-expanded`}>
          <td colSpan={6} className="p-0">
            <ExpandedFolds modelVariant={model.model_variant} numFolds={model.num_folds} foldResults={model.fold_results} color={color} />
          </td>
        </tr>
      )}
    </>
  );
}

function ExpandedFolds({
  modelVariant,
  numFolds,
  foldResults,
  color,
}: {
  modelVariant: string;
  numFolds: number;
  foldResults: { fold: number; best_f1: number | null; best_accuracy: number | null; best_epoch: number | null }[];
  color: string;
}) {
  const allFolds = useQuery({
    queryKey: ["comparison", "folds", modelVariant],
    queryFn: async () => {
      const results = await Promise.all(
        Array.from({ length: numFolds }, (_, i) => fetchComparisonFold(modelVariant, i))
      );
      return results;
    },
    staleTime: 5 * 60 * 1000,
  });

  const folds = allFolds.data;

  return (
    <div className="bg-muted/10 border-t border-border/30 p-5 space-y-4">
      {allFolds.isLoading ? (
        <div className="flex justify-center py-6">
          <div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
        </div>
      ) : allFolds.error ? (
        <div className="text-xs text-muted-foreground text-center py-4">Failed to load fold details</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
          {folds?.map((fold, i) => (
            <FoldCard key={i} fold={fold} result={foldResults[i]} color={color} index={i} />
          ))}
        </div>
      )}
      <div className="text-[11px] text-muted-foreground text-center tracking-wider uppercase">
        Fold-by-fold breakdown
      </div>
    </div>
  );
}

function FoldCard({
  fold,
  result,
  color,
  index,
}: {
  fold: FoldData;
  result: { fold: number; best_f1: number | null; best_accuracy: number | null; best_epoch: number | null } | undefined;
  color: string;
  index: number;
}) {
  const epochPoints = foldToEpochPoints(fold);
  const bestF1 = result?.best_f1 ?? fold.best_f1;

  return (
    <div className="rounded-xl border border-border/40 bg-background p-3.5 space-y-2.5">
      <div className="flex items-center justify-between">
        <span className="font-bold text-xs text-muted-foreground uppercase tracking-wider">
          Fold {index}
        </span>
        <span className="text-xs font-bold tabular-nums" style={{ color }}>
          F1: {(bestF1 * 100).toFixed(1)}%
        </span>
      </div>

      <div className="h-14">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={epochPoints} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
            <YAxis domain={["auto", "auto"]} hide />
            <Tooltip
              contentStyle={{ fontSize: 11, padding: "4px 8px", borderRadius: 6 }}
              labelFormatter={(v) => `Epoch ${v}`}
              formatter={(v) => [(Number(v) * 100).toFixed(1) + "%", "F1"]}
            />
            <Line
              type="monotone"
              dataKey="val_f1"
              stroke={color}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex justify-between text-[11px] text-muted-foreground tabular-nums">
        <span>Best ep: {result?.best_epoch ?? fold.best_epoch}</span>
        <span>Acc: {((result?.best_accuracy ?? fold.best_accuracy) * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

function CommentLine({ text }: { text: string }) {
  return <p className="pl-4 border-l-2 border-primary/30">{text}</p>;
}
