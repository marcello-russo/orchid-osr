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
  optimizer?: string;
  effective_learning_rate?: number;
  weight_decay?: number;
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

// Classification types

export interface TopPrediction {
  species: string;
  confidence: number;
}

export interface ClassifyUploadResponse {
  top_predictions: TopPrediction[];
  predicted_species: string;
  confidence: number;
}

export interface ExplainResponse {
  heatmap_base64: string;
  predicted_species: string;
  confidence: number;
}

// Species / Dataset types

export interface SpeciesInfo {
  name: string;
  image_count: number;
}

export interface SpeciesListResponse {
  species: SpeciesInfo[];
  total_images: number;
  total_species: number;
}

// Model selection

export interface ModelOption {
  id: string;
  name: string;
  model_variant: string;
  mean_f1: number | null;
  test_accuracy: number | null;
  num_folds: number | null;
}

// Dataset types

export interface DatasetClassInfo {
  name: string;
  train_count: number;
  test_count: number;
  total: number;
}

export interface DatasetClassesResponse {
  classes: DatasetClassInfo[];
  total_train: number;
  total_test: number;
}

export interface DatasetImage {
  filename: string;
  url: string;
}

export interface ClassImagesResponse {
  class_name: string;
  split: string;
  total: number;
  images: DatasetImage[];
}

// Comparison types

export interface FoldSnapshot {
  fold: number;
  best_f1: number | null;
  best_accuracy: number | null;
  best_epoch: number | null;
}

export interface ModelComparisonEntry {
  name: string;
  model_variant: string;
  folds_completed: number;
  num_folds: number;
  mean_f1: number | null;
  std_f1: number | null;
  mean_accuracy: number | null;
  test_accuracy: number | null;
  test_f1: number | null;
  best_fold: number | null;
  rank: number | null;
  fold_results: FoldSnapshot[];
}

export interface ModelComparisonResponse {
  models: ModelComparisonEntry[];
}

export interface ExperimentModelEntry {
  variant: string;
  display_name: string;
  mean_f1: number | null;
  std_f1: number | null;
  mean_accuracy: number | null;
  test_accuracy: number | null;
}

export interface ExperimentGroup {
  slug: string;
  label: string;
  description: string;
  has_leakage: boolean;
  models: Record<string, ExperimentModelEntry>;
}

export interface ExperimentComparisonResponse {
  groups: ExperimentGroup[];
  architectures: string[];
}

export interface SeriesModelEntry {
  variant: string;
  display_name: string;
  f1_ce: number | null;
  f1_supcon: number | null;
  delta: number | null;
}

export interface ExperimentComparisonResponse {
  exp6: ModelComparisonEntry[];
  exp7: ModelComparisonEntry[];
  delta: SeriesModelEntry[];
}

export interface EmbeddingPoint {
  idx: number;
  x: number;
  y: number;
  z: number;
  species: string;
  filename: string;
  image_url: string;
  split: string;
}

export interface ServingInfo {
  note: string;
  repo: string;
  path: string;
}

export interface EmbeddingResponse {
  experiment: string;
  method: string;
  dimension: number;
  total_points: number;
  species: string[];
  points: EmbeddingPoint[];
  serving: ServingInfo;
}

// --- NCD types ---

export interface NcdClusteringResult {
  k?: number;
  acc?: number;
  nmi?: number;
  ari?: number;
  ami?: number;
  f_measure?: number;
  silhouette?: number;
  n_clusters?: number;
  n_noise?: number;
  selection_criterion?: string;
}

export interface NcdOsrPerClass {
  left_out: string;
  auroc: number;
  known_acc: number;
  unknown_recall: number;
}

export interface NcdOsrResults {
  mean_auroc: number;
  oscr_curve: { reject_rate: number; known_acc: number }[];
  per_class: NcdOsrPerClass[];
  baselines?: {
    softmax_threshold: { mean_auroc: number };
  };
}

export interface NcdResults {
  model_variant: string;
  exp: number;
  clustering: Record<string, NcdClusteringResult>;
  openset: NcdOsrResults;
}

export interface NcdComparisonEntry {
  experiment: string;
  exp_prefix: string;
  model: string;
  selected_k: number;
  acc: number;
  nmi: number;
  ari: number;
  ami: number;
  f_measure: number;
  silhouette: number;
  mean_auroc: number;
  knn_auroc: number;
  baseline_auroc: number;
  improvement: number;
}

export interface NcdComparison {
  models: NcdComparisonEntry[];
}
