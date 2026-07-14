import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { FoldPhase } from "@/lib/types";

interface PhaseSectionProps {
  phase: FoldPhase;
  epochs: number;
}

const COLORS = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6", "#EF4444"];

export function PhaseSection({ phase, epochs }: PhaseSectionProps) {
  const { t } = useTranslation();
  if (phase.type !== "contrastive") return null;

  const chartData = useMemo(() => {
    const train = phase.metrics["train_loss"] ?? [];
    const knn = phase.metrics["val_knn_acc"] ?? [];
    return Array.from({ length: epochs }, (_, i) => ({
      epoch: i + 1,
      train_loss: train[i],
      val_knn_acc: knn[i],
    }));
  }, [phase, epochs]);

  return (
    <div className="space-y-4 mb-6">
      <div className="flex items-center gap-3">
        <div className="flex size-8 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-500">
          <span className="text-xs font-bold">P1</span>
        </div>
        <div>
          <h3 className="text-sm font-semibold">{phase.name}</h3>
          <p className="text-xs text-muted-foreground">{epochs} epochs</p>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">SupCon Training Loss</CardTitle>
        </CardHeader>
        <CardContent className="pt-2">
          <div style={{ width: "100%", height: 200 }}>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
                <Tooltip />
                <Line type="monotone" dataKey="train_loss" stroke={COLORS[0]} dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">KNN Validation Accuracy</CardTitle>
        </CardHeader>
        <CardContent className="pt-2">
          <div style={{ width: "100%", height: 200 }}>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="val_knn_acc" stroke={COLORS[1]} dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
