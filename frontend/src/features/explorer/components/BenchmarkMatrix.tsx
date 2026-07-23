import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { ExperimentSummary } from "@/lib/types";
import {
  getExperimentSeries, getArchitectureFamily,
  ARCHITECTURE_ORDER, LABEL_TO_VARIANT,
} from "@/features/explorer/types";
import { f1Color, aurocColor } from "@/features/explorer/utils/metrics-colors";
import { MatrixSection } from "./MatrixSection";

interface BenchmarkMatrixProps {
  experiments: ExperimentSummary[];
  familyFilter: string | null;
  seriesFilter: string | null;
}

const MATRIX_SERIES = ["Exp 6", "Exp 7", "Exp 8", "Exp 10"] as const;
const OSR_SERIES = ["Exp 8", "Exp 10"] as const;

export function BenchmarkMatrix({ experiments, familyFilter, seriesFilter }: BenchmarkMatrixProps) {
  const { t } = useTranslation();

  const lookup = useMemo(() => {
    const map: Record<string, Record<string, ExperimentSummary>> = {};
    for (const exp of experiments) {
      const v = exp.model_variant;
      const s = getExperimentSeries(exp.id);
      if (!v || !s) continue;
      const existing = map[v]?.[s];
      if (existing) {
        if (existing.folds_completed !== exp.folds_completed) {
          if ((existing.folds_completed ?? 0) > (exp.folds_completed ?? 0)) continue;
        } else if ((existing.timestamp ?? "") > (exp.timestamp ?? "")) continue;
        else if (existing.id < exp.id) continue;
      }
      if (!map[v]) map[v] = {};
      map[v][s] = exp;
    }
    return map;
  }, [experiments]);

  const visibleSeries = useMemo(() =>
    seriesFilter ? [seriesFilter] : [...MATRIX_SERIES],
  [seriesFilter]);

  const visibleOSR = useMemo(() =>
    seriesFilter
      ? (OSR_SERIES.includes(seriesFilter as typeof OSR_SERIES[number]) ? [seriesFilter] : [])
      : [...OSR_SERIES],
  [seriesFilter]);

  const architectureRows = useMemo(() =>
    ARCHITECTURE_ORDER.filter((archLabel) => {
      if (!familyFilter) return true;
      const variant = LABEL_TO_VARIANT[archLabel];
      const family = variant ? getArchitectureFamily(variant) : null;
      return family === familyFilter;
    }),
  [familyFilter]);

  const f1Best = useMemo(() => {
    const best: Record<string, number> = {};
    for (const col of visibleSeries) {
      best[col] = Math.max(
        ...architectureRows.map((arch) => {
          const exp = lookup[LABEL_TO_VARIANT[arch]]?.[col];
          return exp?.mean_f1 ?? 0;
        }),
      );
    }
    return best;
  }, [lookup, visibleSeries, architectureRows]);

  const osrBest = useMemo(() => {
    const best: Record<string, number> = {};
    for (const col of visibleOSR) {
      best[col] = Math.max(
        ...architectureRows.map((arch) => {
          const exp = lookup[LABEL_TO_VARIANT[arch]]?.[col];
          return exp?.ncd_auroc ?? 0;
        }),
      );
    }
    return best;
  }, [lookup, visibleOSR, architectureRows]);

  return (
    <div className="space-y-8">
      <MatrixSection
        lookup={lookup}
        architectureRows={architectureRows}
        columns={visibleSeries}
        accessor={(e) => e.mean_f1}
        colorFn={f1Color}
        bestPerColumn={f1Best}
        tooltipLabel={t("experiments.matrix.f1_tooltip")}
      />
      {visibleOSR.length > 0 && (
        <MatrixSection
          lookup={lookup}
          architectureRows={architectureRows}
          columns={visibleOSR}
          title={t("experiments.matrix.osr_title")}
          accessor={(e) => e.ncd_auroc}
          colorFn={aurocColor}
          bestPerColumn={osrBest}
          tooltipLabel={t("experiments.matrix.auroc_tooltip")}
        />
      )}
    </div>
  );
}
