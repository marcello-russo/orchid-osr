import { useState, useMemo } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  ComposedChart, LineChart, BarChart, Bar, Area, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { ExperimentDetail, FoldData } from "@/lib/types";
import { foldToEpochPoints } from "../types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ConfusionMatrixChart } from "./ConfusionMatrixChart";
import { cn } from "@/lib/utils";

interface Props {
  experiments: ExperimentDetail[];
}

interface EpochRow {
  epoch: number;
  [key: string]: number;
}

function processFolds(folds: FoldData[], field: string): { means: number[]; stds: number[]; maxEpochs: number } {
  if (!folds.length) return { means: [], stds: [], maxEpochs: 0 };
  const all = folds.map((f) => foldToEpochPoints(f).map((p) => {
    if (field === "val_loss") return p.val_loss ?? 0;
    if (field === "train_loss") return p.train_loss ?? 0;
    if (field === "val_f1") return p.val_f1 ?? 0;
    if (field === "val_accuracy") return p.val_accuracy ?? 0;
    return 0;
  }));
  const maxEpochs = Math.max(...all.map((a) => a.length));
  const means: number[] = [];
  const stds: number[] = [];
  for (let ep = 0; ep < maxEpochs; ep++) {
    const vals = all.map((a) => a[ep] ?? 0).filter((v) => !isNaN(v));
    const mean = vals.length > 0 ? vals.reduce((s, v) => s + v, 0) / vals.length : 0;
    means.push(mean);
    const variance = vals.length > 1 ? vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length : 0;
    stds.push(Math.sqrt(variance));
  }
  return { means, stds, maxEpochs };
}

const COLORS = ["#3B82F6", "#D97706", "#10B981", "#8B5CF6"];

export function CompareModeView({ experiments }: Props) {
  const expColors = useMemo(() =>
    experiments.map((e, i) => ({ id: e.id, color: COLORS[i % COLORS.length] })),
    [experiments]
  );

  const epochData = useMemo(() => {
    const allExps = experiments.map((exp) => ({
      id: exp.id,
      fl: processFolds(exp.folds, "val_loss"),
      tr: processFolds(exp.folds, "train_loss"),
      f1: processFolds(exp.folds, "val_f1"),
      acc: processFolds(exp.folds, "val_accuracy"),
    }));
    const maxEpochs = Math.max(...allExps.map((e) => e.fl.maxEpochs));
    const rows: EpochRow[] = [];
    for (let ep = 0; ep < maxEpochs; ep++) {
      const row: EpochRow = { epoch: ep + 1 };
      for (const e of allExps) {
        if (ep < e.fl.means.length) {
          row[`${e.id}_vloss`] = e.fl.means[ep];
          row[`${e.id}_tloss`] = e.tr.means[ep];
          const vl_bot = Math.max(0, e.fl.means[ep] - e.fl.stds[ep]);
          row[`${e.id}_vloss_bottom`] = vl_bot;
          row[`${e.id}_vloss_band`] = e.fl.means[ep] + e.fl.stds[ep] - vl_bot;
          const f1_bot = Math.max(0, e.f1.means[ep] - e.f1.stds[ep]);
          const f1_top = Math.min(1, e.f1.means[ep] + e.f1.stds[ep]);
          row[`${e.id}_f1`] = e.f1.means[ep];
          row[`${e.id}_f1_bottom`] = f1_bot;
          row[`${e.id}_f1_band`] = f1_top - f1_bot;
          const acc_bot = Math.max(0, e.acc.means[ep] - e.acc.stds[ep]);
          const acc_top = Math.min(1, e.acc.means[ep] + e.acc.stds[ep]);
          row[`${e.id}_acc`] = e.acc.means[ep];
          row[`${e.id}_acc_bottom`] = acc_bot;
          row[`${e.id}_acc_band`] = acc_top - acc_bot;
        }
      }
      rows.push(row);
    }
    return rows;
  }, [experiments]);

  const maxFolds = Math.max(...experiments.map((e) => e.folds.length));
  const foldData = useMemo(() => {
    const data: Record<string, number | string>[] = [];
    for (let fi = 0; fi < maxFolds; fi++) {
      const pt: Record<string, number | string> = { fold: `Fold ${fi}` };
      for (const exp of experiments) {
        pt[exp.id] = exp.folds[fi]?.best_f1 ?? 0;
      }
      data.push(pt);
    }
    return data;
  }, [experiments, maxFolds]);

  const overfitData = useMemo(() => {
    const allExps = experiments.map((exp) => {
      const allEps = exp.folds.map((f) => foldToEpochPoints(f));
      const maxE = Math.max(...allEps.map((a) => a.length));
      const means: number[] = [];
      for (let ep = 0; ep < maxE; ep++) {
        const ratios = allEps.map((a) => a[ep]?.overfitting_ratio ?? 1).filter((v) => !isNaN(v));
        means.push(ratios.length > 0 ? ratios.reduce((s, v) => s + v, 0) / ratios.length : 1);
      }
      return { id: exp.id, means };
    });
    const maxEpochs = Math.max(...allExps.map((e) => e.means.length));
    const rows: EpochRow[] = [];
    for (let ep = 0; ep < maxEpochs; ep++) {
      const row: EpochRow = { epoch: ep + 1 };
      for (const e of allExps) {
        row[`${e.id}_ratio`] = e.means[ep] ?? 1;
      }
      rows.push(row);
    }
    return rows;
  }, [experiments]);

  const configTable = useMemo(() => {
    const fields: { key: string; label: string }[] = [
      { key: "model_variant", label: "Model" },
      { key: "num_epochs", label: "Epochs" },
      { key: "batch_size", label: "Batch" },
      { key: "optimizer", label: "Optimizer" },
      { key: "learning_rate", label: "LR" },
      { key: "weight_decay", label: "Weight Decay" },
      { key: "patience", label: "Patience" },
      { key: "num_folds", label: "Folds" },
    ];
    const metricFields: { key: "mean_f1" | "mean_accuracy" | "test_accuracy"; label: string }[] = [
      { key: "mean_f1", label: "F1 Score" },
      { key: "mean_accuracy", label: "Accuracy" },
      { key: "test_accuracy", label: "Test Accuracy" },
    ];
    return { fields, metricFields };
  }, []);

  // Compute global nMax across all experiments (max test images per class)
  const globalNMax = useMemo(() => {
    let mx = 0;
    for (const exp of experiments) {
      const cm = exp.results?.test_metrics?.confusion_matrix;
      if (cm) {
        for (const row of cm) {
          mx = Math.max(mx, row.reduce((s, v) => s + v, 0));
        }
      }
    }
    return mx || 50;
  }, [experiments]);

  const [active, setActive] = useState(0);
  const chartTitles = ["Loss", "Metrics", "Folds", "Overfitting", "Confusion"];
  const [matrixExp, setMatrixExp] = useState(experiments[0]?.id ?? "");
  const clamp = (i: number) => Math.max(0, Math.min(chartTitles.length - 1, i));

  return (
    <div className="space-y-6">
      {/* Config table */}
      <Card>
        <CardHeader className="pb-2 border-b border-border/50">
          <CardTitle className="text-base">Config comparison</CardTitle>
        </CardHeader>
        <CardContent className="pt-0 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/30 text-xs uppercase tracking-widest text-muted-foreground">
                <th className="text-left py-2.5 px-3 font-bold">Parameter</th>
                {expColors.map((ec) => (
                  <th key={ec.id} className="text-center py-2.5 px-3 font-bold" style={{ color: ec.color }}>{ec.id}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/20">
              {configTable.fields.map((f) => (
                <tr key={f.key} className="hover:bg-muted/20">
                  <td className="py-2 px-3 font-medium text-muted-foreground">{f.label}</td>
                  {experiments.map((exp) => (
                    <td key={exp.id} className="text-center py-2 px-3 tabular-nums">
                      {String((exp.config as any)[f.key] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
              {configTable.metricFields.map((mf) => {
                const vals = experiments.map((exp) => exp.summary[mf.key] as number | null | undefined);
                const best = Math.max(...vals.filter((v): v is number => v !== null && v !== undefined));
                return (
                  <tr key={mf.key} className="hover:bg-muted/20">
                    <td className="py-2 px-3 font-medium text-muted-foreground">{mf.label}</td>
                    {vals.map((v, i) => (
                      <td key={experiments[i].id} className={cn("text-center py-2 px-3 tabular-nums", v === best && "font-bold")}>
                        {v != null ? `${(v * 100).toFixed(2)}%` : "—"}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Carousel */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 overflow-x-auto scrollbar-none">
          <Button variant="ghost" size="icon-sm" onClick={() => setActive((a) => clamp(a - 1))}
            disabled={active === 0} className="shrink-0 hidden sm:flex">
            <ChevronLeft className="size-4" />
          </Button>
          {chartTitles.map((title, i) => (
            <button key={i} onClick={() => setActive(i)}
              className={cn(
                "shrink-0 rounded-full px-4 py-2 text-sm font-medium transition-colors whitespace-nowrap min-h-[44px] cursor-pointer",
                i === active ? "bg-primary text-primary-foreground" : "text-muted-foreground bg-muted/50 hover:bg-muted"
              )}>
              {title}
            </button>
          ))}
          <Button variant="ghost" size="icon-sm" onClick={() => setActive((a) => clamp(a + 1))}
            disabled={active >= chartTitles.length - 1} className="shrink-0 hidden sm:flex">
            <ChevronRight className="size-4" />
          </Button>
        </div>

        {active === 0 && <LossChart epochData={epochData} expColors={expColors} />}
        {active === 1 && <MetricsChart epochData={epochData} expColors={expColors} />}
        {active === 2 && <FoldsChart foldData={foldData} expColors={expColors} />}
        {active === 3 && <OverfitChart overfitData={overfitData} expColors={expColors} />}
        {active === 4 && (
          <Card>
            <CardHeader className="pb-2 border-b border-border/50">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Confusion Matrix</CardTitle>
                <select value={matrixExp} onChange={(e) => setMatrixExp(e.target.value)}
                  className="rounded-md border border-border/50 bg-background px-3 py-1 text-sm">
                  {experiments.map((e) => (
                    <option key={e.id} value={e.id}>{e.id}</option>
                  ))}
                </select>
              </div>
            </CardHeader>
            <CardContent className="pt-4">
              {(() => {
                const sel = experiments.find((e) => e.id === matrixExp);
                const cm = sel?.results?.test_metrics;
                return cm
                  ? <ConfusionMatrixChart matrix={cm.confusion_matrix} labels={cm.confusion_matrix_labels} nMax={globalNMax} />
                  : <p className="text-center text-muted-foreground text-sm py-12">Not available for {matrixExp}.</p>;
              })()}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

/* ---- Sub-charts ---- */

function LossChart({ epochData, expColors }: { epochData: EpochRow[]; expColors: { id: string; color: string }[] }) {
  const [mode, setMode] = useState<"train" | "val">("val");
  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Loss</CardTitle>
          <div className="flex gap-1">
            {(["train", "val"] as const).map((m) => (
              <button key={m} onClick={() => setMode(m)}
                className={cn("rounded-md px-2 py-0.5 text-xs font-medium transition-colors cursor-pointer",
                  mode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground bg-muted/50 hover:bg-muted"
                )}>{m === "train" ? "Train" : "Validation"}</button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={epochData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis dataKey="epoch" tick={{ fontSize: 13 }} />
            <YAxis tick={{ fontSize: 13 }} />
            <Tooltip formatter={(v) => Number(v).toFixed(4)} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {expColors.map((ec) => {
              if (mode === "val") {
                return [
                  <Area key={`${ec.id}_lbot`} type="monotone" dataKey={`${ec.id}_vloss_bottom`} stackId={ec.id}
                    fill="transparent" stroke="none" isAnimationActive={false} />,
                  <Area key={`${ec.id}_lband`} type="monotone" dataKey={`${ec.id}_vloss_band`} stackId={ec.id}
                    fill={ec.color} fillOpacity={0.1} stroke="none" isAnimationActive={false} />,
                  <Line key={`${ec.id}_lline`} type="monotone" dataKey={`${ec.id}_vloss`} name={ec.id}
                    stroke={ec.color} strokeWidth={2} dot={false} isAnimationActive={false} />,
                ];
              }
              return (
                <Line key={`${ec.id}_lline`} type="monotone" dataKey={`${ec.id}_tloss`} name={ec.id}
                  stroke={ec.color} strokeWidth={2} dot={false} isAnimationActive={false} />
              );
            })}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function MetricsChart({ epochData, expColors }: { epochData: EpochRow[]; expColors: { id: string; color: string }[] }) {
  const [metric, setMetric] = useState<"f1" | "accuracy">("f1");
  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Metrics</CardTitle>
          <div className="flex gap-1">
            {(["f1", "accuracy"] as const).map((m) => (
              <button key={m} onClick={() => setMetric(m)}
                className={cn("rounded-md px-2 py-0.5 text-xs font-medium transition-colors cursor-pointer",
                  metric === m ? "bg-primary text-primary-foreground" : "text-muted-foreground bg-muted/50 hover:bg-muted"
                )}>{m === "f1" ? "F1" : "Accuracy"}</button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={epochData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis dataKey="epoch" tick={{ fontSize: 13 }} />
            <YAxis tick={{ fontSize: 13 }} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(2)}%`} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {expColors.map((ec) => {
              const dk = metric === "f1" ? `${ec.id}_f1` : `${ec.id}_acc`;
              const bk = metric === "f1" ? `${ec.id}_f1_bottom` : `${ec.id}_acc_bottom`;
              const bandk = metric === "f1" ? `${ec.id}_f1_band` : `${ec.id}_acc_band`;
              return [
                <Area key={`${ec.id}_mbot`} type="monotone" dataKey={bk} stackId={ec.id}
                  fill="transparent" stroke="none" isAnimationActive={false} />,
                <Area key={`${ec.id}_mband`} type="monotone" dataKey={bandk} stackId={ec.id}
                  fill={ec.color} fillOpacity={0.1} stroke="none" isAnimationActive={false} />,
                <Line key={`${ec.id}_mline`} type="monotone" dataKey={dk} name={ec.id}
                  stroke={ec.color} strokeWidth={2} dot={false} isAnimationActive={false} />,
              ];
            })}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function FoldsChart({ foldData, expColors }: { foldData: Record<string, number | string>[]; expColors: { id: string; color: string }[] }) {
  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-base">Fold Summary</CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={foldData} margin={{ top: 20, right: 20, left: 0, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis dataKey="fold" tick={{ fontSize: 13 }} />
            <YAxis tick={{ fontSize: 13 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} />
            <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(2)}%`} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {expColors.map((ec) => (
              <Bar key={ec.id} dataKey={ec.id} name={ec.id} fill={ec.color}
                radius={[2, 2, 0, 0]} opacity={0.85} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function OverfitChart({ overfitData, expColors }: { overfitData: EpochRow[]; expColors: { id: string; color: string }[] }) {
  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-base">Overfitting Proxy</CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={overfitData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
            <XAxis dataKey="epoch" tick={{ fontSize: 13 }} />
            <YAxis tick={{ fontSize: 13 }} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <ReferenceLine y={1} stroke="var(--border)" strokeDasharray="4 4" />
            {expColors.map((ec) => (
              <Line key={ec.id} type="monotone" dataKey={`${ec.id}_ratio`} name={ec.id}
                stroke={ec.color} strokeWidth={2} dot={false} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
