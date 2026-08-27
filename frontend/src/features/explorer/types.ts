// frontend/src/features/explorer/types.ts

import type {
  ExperimentSummary,
  ExperimentDetail,
  FoldData,
} from "@/lib/types";

export type { ExperimentSummary, ExperimentDetail, FoldData };

/** Epoch data point for line charts */
export interface EpochPoint {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  val_f1?: number;
  val_accuracy?: number;
  overfitting_ratio?: number;
}

/** Convert fold history arrays to chart-friendly array of objects */
export function foldToEpochPoints(fold: FoldData): EpochPoint[] {
  const len = fold.history.train_loss.length;
  const points: EpochPoint[] = [];
  for (let i = 0; i < len; i++) {
    points.push({
      epoch: i + 1,
      train_loss: fold.history.train_loss[i],
      val_loss: fold.history.val_loss[i],
      val_f1: fold.history.val_f1[i],
      val_accuracy: fold.history.val_accuracy[i],
      overfitting_ratio:
        fold.history.train_loss[i] !== 0
          ? fold.history.val_loss[i] / fold.history.train_loss[i]
          : undefined,
    });
  }
  return points;
}

// --- Experiments page redesign helpers ---

export type ArchitectureFamily = "ResNet" | "ConvNeXt" | "DINOv2";
export type ExperimentSeries = "Exp 1" | "Exp 2" | "Exp 3" | "Exp 4";

const FAMILY_MAP: Record<string, ArchitectureFamily> = {
  resnet18: "ResNet", resnet50: "ResNet",
  convnext_tiny: "ConvNeXt", convnext_small: "ConvNeXt",
  dinov2_small: "DINOv2", dinov2_base: "DINOv2",
};

const SERIES_MAP: Record<string, ExperimentSeries> = {
  exp1: "Exp 1", exp2: "Exp 2", exp3: "Exp 3", exp4: "Exp 4",
};

export function getArchitectureFamily(variant: string | null): ArchitectureFamily | null {
  return variant ? (FAMILY_MAP[variant] ?? null) : null;
}

export function getExperimentSeries(id: string): ExperimentSeries | null {
  const prefix = id.split("_")[0];
  return SERIES_MAP[prefix] ?? null;
}

export const FAMILY_COLORS: Record<ArchitectureFamily, string> = {
  ResNet: "#3B82F6",
  ConvNeXt: "#D97706",
  DINOv2: "#8B5CF6",
};

export const ARCHITECTURE_ORDER = [
  "ResNet-18", "ResNet-50",
  "ConvNeXt-Tiny", "ConvNeXt-Small",
  "DINOv2-Small", "DINOv2-Base",
];

export function getArchitectureLabel(variant: string | null): string {
  const map: Record<string, string> = {
    resnet18: "ResNet-18", resnet50: "ResNet-50",
    convnext_tiny: "ConvNeXt-Tiny", convnext_small: "ConvNeXt-Small",
    dinov2_small: "DINOv2-Small", dinov2_base: "DINOv2-Base",
  };
  return variant ? (map[variant] ?? variant) : "—";
}

export const LABEL_TO_VARIANT: Readonly<Record<string, string>> = {
  "ResNet-18": "resnet18", "ResNet-50": "resnet50",
  "ConvNeXt-Tiny": "convnext_tiny", "ConvNeXt-Small": "convnext_small",
  "DINOv2-Small": "dinov2_small", "DINOv2-Base": "dinov2_base",
};