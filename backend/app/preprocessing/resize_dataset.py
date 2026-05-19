"""
Produce architecture-specific resized datasets from cropped images.

Reads from a cropped directory and produces letterbox-resized versions
for each target architecture size, derived from the model registry.
"""
import logging
from pathlib import Path

from PIL import Image

from app.preprocessing.resizing import letterbox_resize
from app.preprocessing.rotation import is_landscape, rotate_image

logger = logging.getLogger(__name__)


def resize_dataset(input_dir, output_dir, target_width, target_height):
    """Resize all images in input_dir to target dimensions via letterbox.

    Applies rotate-if-landscape before resizing. Preserves class subdirectory
    structure.

    Returns:
        Summary dict {total, resized, errors}.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    summary = {"total": 0, "resized": 0, "errors": 0}

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        logger.warning(f"No class directories in {input_dir}")
        return summary

    for class_dir in class_dirs:
        out_class = output_dir / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in class_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")])

        for img_path in files:
            summary["total"] += 1
            try:
                img = Image.open(img_path).convert("RGB")
                if is_landscape(img):
                    img = rotate_image(img)
                img = letterbox_resize(img, target_width, target_height)
                img.save(out_class / img_path.name, quality=95)
                summary["resized"] += 1
            except Exception as e:
                logger.error(f"Resize failed: {img_path}: {e}")
                summary["errors"] += 1

        logger.info(f"{class_dir.name}: {len(files)} → {out_class}")

    logger.info(f"Resize {target_width}×{target_height}: "
                 f"{summary['resized']}/{summary['total']}")
    return summary


def resize_for_all_architectures(input_dir, base_dir, split_name=None):
    """Produce resized datasets for all registered architecture sizes.

    Reads unique input sizes from the model registry and creates
    one output directory per size. If split_name is given (e.g. "train_clean"),
    output goes to cropped_{W}x{H}/{split_name}/{class}/.

    Args:
        input_dir: Cropped dataset root (e.g. dataset/cropped/train_clean/).
        base_dir: Parent for output directories (e.g. dataset/).
        split_name: Optional split subdirectory name.

    Returns:
        Dict mapping size label → output directory path.
    """
    from app.core.models.registry import unique_input_sizes

    results = {}
    for label, (w, h) in unique_input_sizes().items():
        out_dir = Path(base_dir) / f"cropped_{label}"
        if split_name:
            out_dir = out_dir / split_name
        logger.info(f"Resizing to {w}×{h} → {out_dir}")
        resize_dataset(input_dir, out_dir, w, h)
        results[label] = str(out_dir)
    return results
