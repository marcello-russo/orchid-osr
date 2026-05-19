"""
Flower cropper using Faster R-CNN.

Loads a pretrained Faster R-CNN detector, runs inference on raw images,
and crops to the detected flower bounding box.
"""
import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

logger = logging.getLogger(__name__)

DEFAULT_SCORE_THRESHOLD = 0.5


def load_cropper_model(model_path, device=None, num_classes=2):
    """Load the Faster R-CNN cropper model.

    Args:
        model_path: Path to .pth state_dict.
        num_classes: Number of classes (default 2 = background + flower).
        device: Target device (auto-detected if None).

    Returns:
        Model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Cropper model loaded from {model_path} → {device}")
    return model


def detect_flower(model, image, score_threshold=DEFAULT_SCORE_THRESHOLD):
    """Detect the most prominent flower in an image.

    Returns:
        Bounding box (x_min, y_min, x_max, y_max) or None.
    """
    tensor = T.ToTensor()(image)
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model([tensor.to(device)])

    outputs = outputs[0]
    if len(outputs["scores"]) == 0:
        return None

    scores = outputs["scores"]
    best_idx = scores.argmax().item()
    if scores[best_idx].item() < score_threshold:
        return None

    box = outputs["boxes"][best_idx].cpu().tolist()
    return tuple(int(round(v)) for v in box)


def crop_to_bbox(image, bbox, padding=0):
    """Crop image to bounding box with optional padding.

    Args:
        bbox: (x_min, y_min, x_max, y_max).
        padding: Extra pixels around bbox (clamped to image bounds).

    Returns:
        Cropped PIL Image.
    """
    x_min, y_min, x_max, y_max = bbox
    iw, ih = image.size

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(iw, x_max + padding)
    y_max = min(ih, y_max + padding)

    return image.crop((x_min, y_min, x_max, y_max))


def crop_single_image(model, image_path, score_threshold=DEFAULT_SCORE_THRESHOLD,
                      padding=0, fallback="full"):
    """Crop a single image using the detector.

    Args:
        fallback: "full" (return full image), "skip" (return None).

    Returns:
        (cropped_image, metadata_dict) or (None, metadata_dict) if skip.
    """
    image = Image.open(image_path).convert("RGB")
    metadata = {"path": image_path, "detected": False, "score": None, "fallback_used": False}

    bbox = detect_flower(model, image, score_threshold)
    if bbox is not None:
        metadata["detected"] = True
        metadata["bbox"] = bbox
        return crop_to_bbox(image, bbox, padding), metadata

    metadata["fallback_used"] = True
    if fallback == "full":
        logger.debug(f"No detection, using full image: {image_path}")
        return image, metadata
    else:  # skip
        return None, metadata


def crop_dataset(input_dir, output_dir, model_path,
                 score_threshold=DEFAULT_SCORE_THRESHOLD, padding=0,
                 fallback="full", device=None, num_classes=2):
    """Crop all images in input_dir and save to output_dir.

    Preserves class subdirectory structure. Processes all jpg/jpeg/png files.

    Returns:
        Summary dict with counts.
    """
    model = load_cropper_model(model_path, device, num_classes)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    summary = {"total": 0, "cropped": 0, "fallback": 0, "failed": 0, "classes": {}}

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        logger.error(f"No class directories found in {input_dir}")
        return summary

    for class_dir in class_dirs:
        class_name = class_dir.name
        out_class_dir = output_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in class_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        class_stats = {"total": len(files), "cropped": 0, "fallback": 0, "failed": 0}

        for i, img_path in enumerate(files):
            summary["total"] += 1
            try:
                cropped, meta = crop_single_image(
                    model, str(img_path), score_threshold, padding, fallback
                )
                if cropped is None:
                    class_stats["failed"] += 1
                    summary["failed"] += 1
                    continue

                out_path = out_class_dir / img_path.name
                cropped.save(out_path, quality=95)

                if meta["detected"]:
                    class_stats["cropped"] += 1
                    summary["cropped"] += 1
                else:
                    class_stats["fallback"] += 1
                    summary["fallback"] += 1

            except Exception as e:
                logger.error(f"Failed: {img_path}: {e}")
                class_stats["failed"] += 1
                summary["failed"] += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                logger.info(f"  {class_name}: {i + 1}/{len(files)}")

        summary["classes"][class_name] = class_stats
        logger.info(f"{class_name}: {class_stats['cropped']} cropped, "
                     f"{class_stats['fallback']} fallback, {class_stats['failed']} failed")

    logger.info(f"Total: {summary['total']} → {summary['cropped']} cropped, "
                 f"{summary['fallback']} fallback, {summary['failed']} failed")
    return summary
