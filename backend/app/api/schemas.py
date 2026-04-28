# app/api/schemas.py
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime


class ClassifyRequest(BaseModel):
    """Request for image classification."""
    image_base64: str
    model_type: str = "resnet18"


class ClassifyResponse(BaseModel):
    """Response for image classification."""
    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: Dict[str, float]


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
    model_variant: Optional[str] = None
    type: str
    status: str
    num_folds: Optional[int] = None
    test_accuracy: Optional[float] = None
    mean_f1: Optional[float] = None
    std_f1: Optional[float] = None
    timestamp: Optional[str] = None


class ExperimentDetailResponse(BaseModel):
    """Full experiment detail."""
    id: str
    config: dict
    folds: list[dict]
    results: Optional[dict] = None
    summary: dict


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


class TopPrediction(BaseModel):
    """A single species prediction with confidence."""
    species: str
    confidence: float


class ClassifyUploadResponse(BaseModel):
    """Response for file-upload classification."""
    top_predictions: list[TopPrediction]
    predicted_species: str
    confidence: float


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
    test_accuracy: float | None = None
    num_folds: int | None = None


class ModelListResponse(BaseModel):
    """List of available classification models."""
    models: list[ModelOption]


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
