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