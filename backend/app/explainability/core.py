"""Core inference and probability utilities."""

import torch
from torch.nn import functional as F


def inference(model, image, device):
    """Run model inference on a single image."""
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        values = model(image)
        _, predicted = torch.max(values, 1)
    return values, predicted


def get_percentages(values):
    """Convert logits to percentage probabilities."""
    return F.softmax(values, dim=1) * 100
