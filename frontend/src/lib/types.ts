// frontend/src/lib/types.ts

export interface ExperimentConfig {
  experiment?: number;
  name?: string;
  train?: string;
  test?: string;
  type?: string;
  model_variant?: string;
  num_folds?: number;
  num_epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  seed?: number;
  num_classes?: number;
  class_names?: string[];
  patience?: number;
  timestamp?: string;
}

export interface FoldData {
  fold: number;
  best_f1: number;
  best_accuracy: number;
  best_epoch: number;
  final_train_loss: number;
  final_val_loss: number;
  history: {
    train_loss: number[];
    val_loss: number[];
    val_accuracy: number[];
    val_f1: number[];
  };
  train_size: number;
  val_size: number;
}

export interface PerClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
  tp: number;
  fp: number;
  fn: number;
}

export interface TestMetrics {
  accuracy: number;
  f1_macro: number;
  precision_macro: number;
  recall_macro: number;
  f1_per_class: Record<string, number>;
  confusion_matrix: number[][];
  confusion_matrix_labels: string[];
  per_class_metrics: Record<string, PerClassMetrics>;
}

export interface ExperimentResults {
  mean_accuracy: number;
  std_accuracy: number;
  mean_f1: number;
  std_f1: number;
  best_fold: number;
  model_variant: string;
  num_folds: number;
  test_metrics: TestMetrics;
  fold_results?: FoldData[];
}

export interface ExperimentSummary {
  id: string;
  name: string;
  model_variant: string | null;
  type: "baseline" | "kfold";
  status: "completed" | "partial";
  num_folds: number | null;
  test_accuracy: number | null;
  mean_f1: number | null;
  std_f1: number | null;
  timestamp: string | null;
}

export interface ExperimentDetail {
  id: string;
  config: ExperimentConfig;
  folds: FoldData[];
  results: ExperimentResults | null;
  summary: {
    model_variant: string | null;
    type: "baseline" | "kfold";
    num_folds: number;
    mean_accuracy: number | null;
    mean_f1: number | null;
    std_f1: number | null;
    test_accuracy: number | null;
    test_f1: number | null;
  };
}
