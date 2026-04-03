# app/api/deps.py
from functools import lru_cache
from pathlib import Path
import logging
import torch
from app.config import settings
from app.core.models.resnet import ResNetConfig, ResNetClassifier

logger = logging.getLogger(__name__)

VALID_MODEL_TYPES = ("resnet18", "resnet50")

# Class names for prediction
CLASS_NAMES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. majellensis",
    "O. sphegodes",
    "O. sphegodes_Palena"
]


@lru_cache()
def get_device() -> torch.device:
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_weights(model_type: str) -> Path | None:
    """Find best available trained weights for a model type."""
    models_dir = Path(settings.models_dir)

    # Prefer best fold weights from k-fold experiments
    for exp_dir in sorted(models_dir.glob("*")):
        if not exp_dir.is_dir():
            continue
        # Check k-fold results for best fold
        for kfold_dir in sorted(exp_dir.glob(f"*kfold_{model_type}")):
            results_json = kfold_dir / "results.json"
            if results_json.exists():
                import json
                with open(results_json) as f:
                    results = json.load(f)
                if "best_model_path" in results:
                    path = Path(results["best_model_path"])
                    if path.exists():
                        logger.info(f"Loading best k-fold weights: {path}")
                        return path

    # Fallback: standard trained model
    for subdir in sorted(models_dir.glob(f"train_*_{model_type}")):
        model_path = subdir / "model.pt"
        if model_path.exists():
            logger.info(f"Loading trained weights: {model_path}")
            return model_path

    return None


@lru_cache()
def get_model(model_type: str = "resnet18") -> ResNetClassifier:
    """
    Load and cache model with trained weights if available.

    Falls back to ImageNet pretrained weights if no trained model found.
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Valid: {VALID_MODEL_TYPES}")

    config = ResNetConfig(
        variant=model_type,
        num_classes=settings.num_classes,
        pretrained=True
    )

    model = ResNetClassifier(config)

    # Try loading trained weights
    weights_path = _find_weights(model_type)
    if weights_path:
        model.load_weights(str(weights_path))
        logger.info(f"Loaded trained weights from {weights_path}")
    else:
        logger.warning(f"No trained weights found for {model_type}, using ImageNet pretrained")

    model.eval()

    device = get_device()
    model = model.to(device)

    return model
