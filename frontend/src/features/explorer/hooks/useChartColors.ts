// frontend/src/features/explorer/hooks/useChartColors.ts
import { useMemo } from "react";
import { useAppStore } from "@/stores/useAppStore";

const CHART_VARS = ["--chart-1", "--chart-2", "--chart-3", "--chart-4", "--chart-5"];
const FALLBACK = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6", "#EF4444"];

export function useChartColors(): string[] {
  const theme = useAppStore((s) => s.theme);
  return useMemo(() => {
    const root = document.documentElement;
    return CHART_VARS.map((v, i) => {
      const val = getComputedStyle(root).getPropertyValue(v).trim();
      return val ? `hsl(${val})` : FALLBACK[i];
    });
  }, [theme]);
}
