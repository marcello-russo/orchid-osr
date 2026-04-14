# app/api/routes/health.py
from fastapi import APIRouter
from datetime import datetime
from app.api.schemas import HealthResponse
from app.api.deps import get_model

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model = get_model()
        model_loaded = True
        model_type = "resnet18"
    except Exception as e:
        model_loaded = False
        model_type = "resnet18"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=model_type,
        timestamp=datetime.now()
    )
