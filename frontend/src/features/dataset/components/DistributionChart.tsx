// frontend/src/features/dataset/components/DistributionChart.tsx
import { useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartSize } from "@/features/explorer/hooks/useChartSize";
import { useChartColors } from "@/features/explorer/hooks/useChartColors";
import type { DatasetClassInfo } from "@/lib/types";

interface DistributionChartProps {
  classes: DatasetClassInfo[];
}

export function DistributionChart({ classes }: DistributionChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { width, height } = useChartSize(ref);
  const chartColors = useChartColors();

  const isMobile = width < 400;
  const data = classes.map((c) => ({
    name: c.name,
    shortName: c.name.replace("O. ", ""),
    Train: c.train_count,
    Test: c.test_count,
  }));

  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-base">Class Distribution</CardTitle>
      </CardHeader>
      <CardContent className="pt-4 px-1 sm:px-6">
        <div ref={ref} style={{ width: "100%", height }}>
          <BarChart 
            data={data} 
            width={width} 
            height={height} 
            layout="vertical" 
            margin={{ left: isMobile ? 0 : 30, right: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis type="number" tick={{ fontSize: 12 }} />
            <YAxis 
              type="category" 
              dataKey={isMobile ? "shortName" : "name"} 
              tick={{ fontSize: 11 }} 
              width={isMobile ? 80 : 120} 
            />
            <Tooltip />
            <Bar dataKey="Train" fill={chartColors[0]} radius={[0, 4, 4, 0]} barSize={12} isAnimationActive={false} />
            <Bar dataKey="Test" fill={chartColors[1]} radius={[0, 4, 4, 0]} barSize={12} isAnimationActive={false} />
          </BarChart>
        </div>
      </CardContent>
    </Card>
  );
}
