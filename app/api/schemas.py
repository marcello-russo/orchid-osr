# app/api/schemas.py
from pydantic import BaseModel
from typing import Dict, Literal
from datetime import datetime


class ClassifyRequest(BaseModel):
    """Request for image classification."""
    image_base64: str
    model_type: Literal["resnet18", "resnet50"] = "resnet18"


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
