# app/api/routes/health.py
from datetime import datetime

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.core.models import available_models

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        models = available_models()
        model_loaded = len(models) > 0
        model_type = ", ".join(models) if models else "none"
    except Exception:
        model_loaded = False
        model_type = "none"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=model_type,
        timestamp=datetime.now(),
    )
