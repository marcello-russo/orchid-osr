// frontend/src/features/explorer/hooks/useChartColors.ts
import { useMemo } from "react";
import { useAppStore } from "@/stores/useAppStore";

const COLORS = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6", "#EF4444"];

export function useChartColors(): string[] {
  const theme = useAppStore((s) => s.theme);
  return useMemo(() => COLORS, [theme]);
}
