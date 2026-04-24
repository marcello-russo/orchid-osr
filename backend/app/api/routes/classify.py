# app/api/routes/classify.py
import base64
import io
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import torch
from PIL import Image
from torchvision import transforms as T

from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyUploadResponse,
    ExplainResponse,
    ModelListResponse,
    TopPrediction,
)
from app.api.deps import get_model, get_device, get_model_options, CLASS_NAMES
from app.preprocessing.normalization import get_mean, get_std

# Image dimensions (same as training pipeline)
IMG_WIDTH = 256
IMG_HEIGHT = 512
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

router = APIRouter()


@router.get("/classify/models", response_model=ModelListResponse)
async def list_models():
    """List available trained models for classification."""
    return ModelListResponse(models=get_model_options())


def _get_preprocess():
    """Build preprocessing pipeline matching training."""
    mean = get_mean()
    std = get_std()
    return T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def _read_upload_image(file: UploadFile) -> tuple[torch.Tensor, Image.Image]:
    """Read an uploaded file into (preprocessed tensor, original PIL Image)."""
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    try:
        image_bytes = file.file.read(MAX_UPLOAD_SIZE + 1)
        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large (max {MAX_UPLOAD_SIZE // 1024 // 1024}MB)")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        preprocess = _get_preprocess()
        tensor = preprocess(pil_image).unsqueeze(0)
        return tensor, pil_image
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


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


@router.post("/classify/upload", response_model=ClassifyUploadResponse)
async def classify_upload(file: UploadFile = File(...), model_id: str = Form(None)):
    """Classify an orchid image uploaded as multipart form data."""
    image_tensor, _ = _read_upload_image(file)

    model = get_model(model_id=model_id) if model_id else get_model("resnet18")
    device = get_device()
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()

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
    )


@router.post("/classify/explain", response_model=ExplainResponse)
async def explain_image(file: UploadFile = File(...), model_id: str = Form(None)):
    """Generate an occlusion-based explainability heatmap for an uploaded image."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.explainability.visualization import (
        compute_occlusion,
        _normalize_attr,
    )
    from app.preprocessing.normalization import denormalize_image, get_mean, get_std

    image_tensor, _ = _read_upload_image(file)

    model = get_model(model_id=model_id) if model_id else get_model("resnet18")
    device = get_device()
    image_tensor = image_tensor.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(dim=1).item()
        confidence = probabilities[0, predicted_idx].item()

    # Occlusion attribution
    sliding_window_size = 15
    stride = 8
    global_attribution = compute_occlusion(
        model, image_tensor.squeeze(0), sliding_window_size, stride
    )

    label_attributions = global_attribution[predicted_idx].squeeze(0).cpu().numpy()
    label_attributions = np.transpose(label_attributions, (1, 2, 0))
    norm_attr = _normalize_attr(label_attributions, sign="all", outlier_perc=2, reduction_axis=2)

    image_np = denormalize_image(
        image_tensor.squeeze(0), get_mean(), get_std()
    ).cpu().numpy().transpose(1, 2, 0)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])

    fig, ax = plt.subplots(figsize=(5, 10))
    ax.imshow(np.mean(image_np, axis=2), cmap="gray")
    ax.imshow(norm_attr, cmap=cmap, vmin=-1, vmax=1, alpha=0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return ExplainResponse(
        heatmap_base64=heatmap_b64,
        predicted_species=CLASS_NAMES[predicted_idx],
        confidence=confidence,
    )
