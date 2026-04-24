// frontend/src/features/explorer/hooks/useChartColors.ts
import { useEffect, useState } from "react";

const FALLBACK = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6", "#EF4444"];

export function useChartColors(): string[] {
  const [colors, setColors] = useState<string[]>(FALLBACK);

  useEffect(() => {
    const root = document.documentElement;
    const vars = ["--chart-1", "--chart-2", "--chart-3", "--chart-4", "--chart-5"];
    const resolved = vars.map((v) => {
      const val = getComputedStyle(root).getPropertyValue(v).trim();
      return val || FALLBACK[vars.indexOf(v)];
    });
    setColors(resolved);
  }, []);

  return colors;
}
