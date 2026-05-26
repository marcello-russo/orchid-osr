# app/api/routes/classify.py
import base64
import io

import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from app.api.deps import (
    CLASS_NAMES, get_device, get_model, get_model_options,
    get_cache_status, download_all_weights, delete_cached_weight, clear_cache,
    require_role,
)
from app.api.schemas import (
    CacheStatus,
    ClassifyRequest,
    ClassifyResponse,
    ClassifyUploadResponse,
    ExplainResponse,
    ModelListResponse,
    TopPrediction,
)
from app.core.models import available_models, create_transform, get_registry_entry

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


@router.get("/classify/cache", response_model=CacheStatus)
async def cache_status(admin=Depends(require_role("admin"))):
    """Get cache status: which models are downloaded, sizes, progress."""
    return get_cache_status()


@router.post("/classify/cache/download")
async def cache_download_all(admin=Depends(require_role("admin"))):
    """Download all missing model weights in the background."""
    download_all_weights(background=True)
    return {"status": "started"}


@router.post("/classify/cache/download/{model_id}")
async def cache_download_one(model_id: str, admin=Depends(require_role("admin"))):
    """Download a single model's weights."""
    from app.api.deps import _resolve_weights
    result = _resolve_weights(model_id)
    if result is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return {"status": "ok", "model_id": model_id}


@router.delete("/classify/cache/{model_id}")
async def cache_delete_one(model_id: str, admin=Depends(require_role("admin"))):
    """Delete a single model's cached weights."""
    deleted = delete_cached_weight(model_id)
    if not deleted:
        raise HTTPException(404, f"No cached weights for {model_id}")
    return {"status": "deleted", "model_id": model_id}


@router.delete("/classify/cache")
async def cache_clear(admin=Depends(require_role("admin"))):
    """Delete all cached model weights."""
    count = clear_cache()
    return {"status": "cleared", "files_removed": count}


@router.post("/classify/explain", response_model=ExplainResponse)
async def explain_image(file: UploadFile = File(...), model_id: str = Form(None), user=Depends(require_role("user"))):
    """Generate an occlusion-based explainability heatmap for an uploaded image."""
    import gc

    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.explainability.visualization import (
        _normalize_attr,
        compute_occlusion,
    )
    from app.preprocessing.normalization import denormalize_image

    plt.close("all")

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
    del outputs, probabilities

    # Occlusion attribution (coarse for CPU performance)
    sliding_window_size = 30
    stride = 20
    global_attribution = compute_occlusion(
        model, image_tensor.squeeze(0), sliding_window_size, stride
    )

    label_attributions = global_attribution[predicted_idx].squeeze(0).cpu().numpy()
    del global_attribution
    label_attributions = np.transpose(label_attributions, (1, 2, 0))
    norm_attr = _normalize_attr(label_attributions, sign="all", outlier_perc=2, reduction_axis=2)
    del label_attributions

    entry = get_registry_entry(model_type)
    image_np = denormalize_image(
        image_tensor.squeeze(0), entry.norm_mean, entry.norm_std
    ).cpu().numpy().transpose(1, 2, 0)
    del image_tensor

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])

    fig, ax = plt.subplots(figsize=(5, 10))
    ax.imshow(np.mean(image_np, axis=2), cmap="gray")
    ax.imshow(norm_attr, cmap=cmap, vmin=-1, vmax=1, alpha=0.5)
    del image_np, norm_attr
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    gc.collect()

    return ExplainResponse(
        heatmap_base64=heatmap_b64,
        predicted_species=CLASS_NAMES[predicted_idx],
        confidence=confidence,
    )
