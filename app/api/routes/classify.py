# app/api/routes/classify.py
import base64
import io
from fastapi import APIRouter, HTTPException
import torch
from PIL import Image
from torchvision import transforms as T

from app.api.schemas import ClassifyRequest, ClassifyResponse
from app.api.deps import get_model, get_device, CLASS_NAMES
from app.preprocessing.normalization import get_mean, get_std

# Image dimensions (same as training pipeline)
IMG_WIDTH = 256
IMG_HEIGHT = 512

router = APIRouter()


def _get_preprocess():
    """Build preprocessing pipeline matching training."""
    mean = get_mean()
    std = get_std()
    return T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def decode_base64_image(image_base64: str) -> torch.Tensor:
    """Decode base64 image to tensor."""
    try:
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        preprocess = _get_preprocess()
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
    model = get_model(request.model_type)
    device = get_device()

    # Decode and preprocess image
    image_tensor = decode_base64_image(request.image_base64)
    image_tensor = image_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()

    # Build response
    probs_dict = {
        CLASS_NAMES[i]: probabilities[0, i].item()
        for i in range(len(CLASS_NAMES))
    }

    return ClassifyResponse(
        predicted_class=CLASS_NAMES[predicted_idx],
        predicted_class_id=predicted_idx,
        confidence=confidence,
        probabilities=probs_dict
    )
