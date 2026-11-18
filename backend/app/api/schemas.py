# app/api/schemas.py
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Request for image classification."""

    image_base64: str = Field(..., max_length=50_000_000)
    model_type: str = "resnet18"


class ClassifyResponse(BaseModel):
    """Response for image classification."""

    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: dict[str, float]
    novelty: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_type: str
    timestamp: datetime


class ExperimentInfo(BaseModel):
    """Experiment information."""

    id: int
    name: str
    train_dataset: str
    test_dataset: str
    status: str


class ExperimentListResponse(BaseModel):
    """List of experiments."""

    experiments: list[ExperimentInfo]


class ExperimentSummary(BaseModel):
    """Summary of an experiment for list view."""

    id: str
    name: str
    model_variant: str | None = None
    type: str
    status: str
    num_folds: int | None = None
    folds_completed: int | None = None
    test_accuracy: float | None = None
    test_f1: float | None = None
    mean_f1: float | None = None
    std_f1: float | None = None
    timestamp: str | None = None
    is_ncd: bool = False
    ncd_auroc: float | None = None
    ncd_aupr: float | None = None
    ncd_detector: str | None = None
    ncd_clustering_nmi: float | None = None


class ExperimentDetailResponse(BaseModel):
    """Full experiment detail."""

    id: str
    config: dict[str, Any]
    folds: list[dict[str, Any]]
    results: dict[str, Any] | None = None
    detectors: dict[str, Any] | None = None
    openset: dict[str, Any] | None = None
    clustering: dict[str, Any] | None = None
    summary: dict[str, Any]


class FoldHistory(BaseModel):
    """Training history arrays for a fold."""

    train_loss: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    val_f1: list[float]


class FoldDataResponse(BaseModel):
    """Full fold detail with training history."""

    fold: int
    best_f1: float
    best_accuracy: float
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    history: FoldHistory
    train_size: int
    val_size: int
    fold_phases: list[dict[str, Any]] | None = None


class TopPrediction(BaseModel):
    """A single species prediction with confidence."""

    species: str
    confidence: float


class ClassifyUploadResponse(BaseModel):
    """Response for file-upload classification."""

    top_predictions: list[TopPrediction]
    predicted_species: str
    confidence: float
    novelty: dict[str, Any] | None = None


class ExplainResponse(BaseModel):
    """Response for explainability heatmap."""

    heatmap_base64: str
    predicted_species: str
    confidence: float


class SpeciesInfo(BaseModel):
    """Info about a single species in the dataset."""

    name: str
    image_count: int


class SpeciesListResponse(BaseModel):
    """List of species with dataset statistics."""

    species: list[SpeciesInfo]
    total_images: int
    total_species: int


class ModelOption(BaseModel):
    """A trained model available for classification."""

    id: str
    name: str
    model_variant: str
    mean_f1: float | None = None
    std_f1: float | None = None
    test_accuracy: float | None = None
    num_folds: int | None = None
    is_cached: bool = False
    size_mb: float | None = None
    exp_num: int = 0
    exp_label: str = ""
    tags: list[str] = []


class ModelListResponse(BaseModel):
    """List of available classification models."""

    models: list[ModelOption]


class CacheStatus(BaseModel):
    """Status of locally cached model weights."""

    total_models: int = 0
    cached_models: int = 0
    total_size_mb: float = 0
    cached_size_mb: float = 0
    is_downloading: bool = False
    download_progress: float = 0
    models: list[ModelOption] = []


class DatasetClassInfo(BaseModel):
    """A single class in the dataset with train/test counts."""

    name: str
    train_count: int
    test_count: int
    total: int


class DatasetClassesResponse(BaseModel):
    """List of dataset classes with distribution stats."""

    classes: list[DatasetClassInfo]
    total_train: int
    total_test: int


class DatasetImage(BaseModel):
    """A single image reference in the dataset."""

    filename: str
    url: str


class ClassImagesResponse(BaseModel):
    """Images for a specific class and split."""

    class_name: str
    split: str
    total: int
    images: list[DatasetImage]


class FoldSnapshot(BaseModel):
    """Single fold result in the comparison view."""

    fold: int
    best_f1: float | None = None
    best_accuracy: float | None = None
    best_epoch: int | None = None


class ModelComparisonEntry(BaseModel):
    """One model's full comparison data."""

    name: str
    model_variant: str
    folds_completed: int
    num_folds: int
    mean_f1: float | None = None
    std_f1: float | None = None
    mean_accuracy: float | None = None
    test_accuracy: float | None = None
    test_f1: float | None = None
    best_fold: int | None = None
    rank: int | None = None
    fold_results: list[FoldSnapshot] = []


class ModelComparisonResponse(BaseModel):
    """Full comparison of all models in the active experiment."""

    models: list[ModelComparisonEntry]


class ExperimentModelEntry(BaseModel):
    """One model's result within an experiment comparison group."""

    variant: str
    display_name: str
    mean_f1: float | None = None
    std_f1: float | None = None
    mean_accuracy: float | None = None
    test_accuracy: float | None = None


class ExperimentGroup(BaseModel):
    """A group of model results from one experiment."""

    slug: str
    label: str
    description: str
    has_leakage: bool = False
    models: dict[str, ExperimentModelEntry] = {}


class SeriesModelEntry(BaseModel):
    """One model's comparison between experiments."""

    variant: str
    display_name: str
    f1_ce: float | None = None
    f1_supcon: float | None = None
    f1_sphor: float | None = None
    delta: float | None = None


class ExperimentComparisonResponse(BaseModel):
    """Comparison across all experiments."""

    exp1: list[ModelComparisonEntry]
    exp2: list[ModelComparisonEntry]
    exp3: list[ModelComparisonEntry]
    delta: list[SeriesModelEntry]


class EmbeddingPoint(BaseModel):
    """A single point in the embedding space."""

    idx: int
    x: float
    y: float
    z: float = 0.0
    species: str
    filename: str = ""
    image_url: str = ""
    split: str = ""


class ServingInfo(BaseModel):
    """Metadata about how embeddings are served."""

    note: str = ""
    repo: str = ""
    path: str = ""


class EmbeddingResponse(BaseModel):
    """Embedding space data for visualization."""

    experiment: str
    method: str
    dimension: int
    total_points: int
    species: list[str]
    points: list[EmbeddingPoint]
    serving: ServingInfo = ServingInfo()


class ErrorResponse(BaseModel):
    error: str
    detail: str
    fields: dict[str, str] | None = None
