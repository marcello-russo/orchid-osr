"""Single-pass embedding + logits extraction per NCD pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from app.core.models import create_transform
from app.core.ncd.extract import load_trained_model
from app.preprocessing.dataset_tool import ImageFolderWithName


@dataclass
class ExtractionBundle:
    train_embeddings: NDArray
    train_logits: NDArray | None
    train_labels: NDArray
    train_class_names: list[str]
    test_embeddings: NDArray
    test_logits: NDArray | None
    test_labels: NDArray
    test_class_names: list[str]


def _get_classifier(model, model_variant: str) -> Callable:
    """Restituisce il classifier layer per l'architettura."""
    if "dinov2" in model_variant:
        return model._head
    elif "resnet" in model_variant:
        return model._backbone.fc
    elif "convnext" in model_variant:
        return model._backbone.classifier[2]
    raise ValueError(f"Unknown model_variant: {model_variant}")


def extract_all(
    model_variant: str,
    train_dir: str,
    test_dir: str,
    model_path: str,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    collect_logits: bool = False,
    exp_dir: str | None = None,
) -> ExtractionBundle:
    """Carica modello e forward UNICO per embeddings + logits."""
    model = load_trained_model(model_variant, num_classes, model_path, device, exp_dir=exp_dir)
    model.eval()
    transform = create_transform(model_variant)

    train_dataset = ImageFolderWithName(train_dir, transform=transform)
    test_dataset = ImageFolderWithName(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    classifier = _get_classifier(model, model_variant) if collect_logits else None

    embeddings = {"train": [], "test": []}
    logits_list = {"train": [], "test": []} if collect_logits else None
    labels = {"train": [], "test": []}

    for split, loader in [("train", train_loader), ("test", test_loader)]:
        for inputs, lbls, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.no_grad():
                emb = model.get_embeddings(inputs)
                embeddings[split].append(emb.cpu().numpy())
                if collect_logits:
                    logit = classifier(emb)
                    logits_list[split].append(logit.cpu().numpy())
            labels[split].append(lbls.numpy())

    return ExtractionBundle(
        train_embeddings=np.concatenate(embeddings["train"]),
        train_logits=np.concatenate(logits_list["train"]) if collect_logits else None,
        train_labels=np.concatenate(labels["train"]),
        train_class_names=train_dataset.classes,
        test_embeddings=np.concatenate(embeddings["test"]),
        test_logits=np.concatenate(logits_list["test"]) if collect_logits else None,
        test_labels=np.concatenate(labels["test"]),
        test_class_names=test_dataset.classes,
    )
