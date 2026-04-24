// frontend/src/features/explorer/hooks/useChartSize.ts
import { useEffect, useState, type RefObject } from "react";

interface ChartSize {
  width: number;
  height: number;
}

const DEFAULT: ChartSize = { width: 800, height: 600 };

function computeHeight(width: number): number {
  if (width < 480) return 250;
  if (width < 640) return 300;
  return 450;
}

export function useChartSize(containerRef: RefObject<HTMLDivElement | null>): ChartSize {
  const [size, setSize] = useState<ChartSize>(DEFAULT);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onResize = () => {
      const w = el.clientWidth;
      if (w > 0) setSize({ width: Math.floor(w), height: computeHeight(w) });
    };

    onResize();
    const observer = new ResizeObserver(onResize);
    observer.observe(el);
    return () => observer.disconnect();
  }, [containerRef]);

  return size;
}
