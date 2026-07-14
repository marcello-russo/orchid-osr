# app/api/routes/classify.py
import base64
import io

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.api.cache import get_model
from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyUploadResponse,
    ModelListResponse,
    TopPrediction,
)
from app.api.weights import CLASS_NAMES, get_device, get_model_options
from app.core.models import available_models, create_transform

ENERGY_TAU = 1.0
ENERGY_THRESHOLD = 0.5  # Calibrated at 10% FPR

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

router = APIRouter()


@router.get("/classify/models", response_model=ModelListResponse)
async def list_models():
    """List available trained models for classification."""
    return ModelListResponse(models=get_model_options())


def _read_upload_image(file: UploadFile, model_type: str) -> tuple[torch.Tensor, Image.Image]:
    """Read an uploaded file into (preprocessed tensor, original PIL Image)."""
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    try:
        image_bytes = file.file.read(MAX_UPLOAD_SIZE + 1)
        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large (max {MAX_UPLOAD_SIZE // 1024 // 1024}MB)")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        preprocess = create_transform(model_type)
        tensor = preprocess(pil_image).unsqueeze(0)
        return tensor, pil_image
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


def decode_base64_image(image_base64: str, model_type: str) -> torch.Tensor:
    """Decode base64 image to tensor."""
    try:
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        preprocess = create_transform(model_type)
        tensor = preprocess(image)
        return tensor.unsqueeze(0)  # Add batch dimension

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(request: ClassifyRequest):
    """
    Classify an orchid image.

    Args:
        request: ClassifyRequest with base64 encoded image

    Returns:
        ClassifyResponse with predicted class and probabilities
    """
    # Get model and device
    model_type = request.model_type
    valid = available_models()
    if model_type not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}. Available: {valid}")
    model = get_model(model_type)
    device = get_device()

    # Decode and preprocess image
    image_tensor = decode_base64_image(request.image_base64, model_type)
    image_tensor = image_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()

    # Build response
    probs_dict = {CLASS_NAMES[i]: probabilities[0, i].item() for i in range(len(CLASS_NAMES))}

    return ClassifyResponse(
        predicted_class=CLASS_NAMES[predicted_idx],
        predicted_class_id=predicted_idx,
        confidence=confidence,
        probabilities=probs_dict,
    )


@router.post("/classify/upload", response_model=ClassifyUploadResponse)
async def classify_upload(file: UploadFile = File(...), model_id: str = Form(None)):
    """Classify an orchid image uploaded as multipart form data."""
    # Load model first to determine model_type for correct transform
    model = get_model(model_id=model_id) if model_id else get_model(available_models()[0])
    model_type = model.model_name

    image_tensor, _ = _read_upload_image(file, model_type)
    device = get_device()
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()

        # Energy-based novelty detection (Liu et al., NeurIPS 2020)
        energy = ENERGY_TAU * torch.logsumexp(outputs / ENERGY_TAU, dim=1).item()
        is_novel = energy > ENERGY_THRESHOLD

    topk = torch.topk(probabilities, min(6, len(CLASS_NAMES)), dim=1)
    top_predictions = [
        TopPrediction(
            species=CLASS_NAMES[topk.indices[0, i].item()],
            confidence=topk.values[0, i].item(),
        )
        for i in range(topk.indices.shape[1])
    ]

    return ClassifyUploadResponse(
        top_predictions=top_predictions,
        predicted_species=CLASS_NAMES[predicted_idx],
        confidence=confidence,
        novelty={
            "is_novel": is_novel,
            "score": round(energy, 4),
            "threshold": ENERGY_THRESHOLD,
        },
    )
