"""Embedding extraction + t-SNE/UMAP dimensionality reduction."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_trained_model(
    model_variant: str,
    num_classes: int,
    best_model_path: str,
    device: torch.device,
    exp_dir: str | None = None,
):
    """Load a trained model from best fold weights.

    If exp_dir is provided, reads config.json to determine model-specific settings
    (e.g., multi_grain for DINOv2).
    """
    from app.core.models import create_model

    model_kwargs = {}
    if exp_dir:
        config_path = os.path.join(exp_dir, "config.json")
        if os.path.exists(config_path):
            import json

            cfg = json.load(open(config_path))
        else:
            cfg = {}
        # Determine multi_grain from experiment dir name or config
        dir_name = os.path.basename(exp_dir)
        if "dinov2" in model_variant:
            is_mgh = "mgh" in dir_name.lower() or cfg.get("multi_grain", False)
            model_kwargs["multi_grain"] = is_mgh
            logger.info(f"Model {model_variant}: multi_grain={is_mgh} (from exp_dir: {dir_name})")
        # SupCon models are trained with L2-normalized embeddings before the
        # linear head (fix for exploding logits / val loss). Match that at load.
        if cfg.get("normalize_embeddings", False):
            model_kwargs["normalize_embeddings"] = True

    model = create_model(model_variant, num_classes=num_classes, pretrained=False, **model_kwargs)
    model.load_weights(best_model_path)
    model = model.to(device)
    model.eval()
    return model


def _create_loader(model_variant: str, data_dir: str, batch_size: int = 8) -> tuple[DataLoader, dict[int, str]]:
    """Create a DataLoader and class mapping for a dataset."""
    from app.core.models import create_transform
    from app.preprocessing.dataset_tool import ImageFolderWithName

    transform = create_transform(model_variant)
    dataset = ImageFolderWithName(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_to_name = {v: k for k, v in dataset.class_to_idx.items()}
    return loader, class_to_name


def extract_embeddings(
    model_variant: str,
    data_dir: str,
    best_model_path: str,
    num_classes: int,
    batch_size: int = 8,
    device: torch.device | None = None,
    exp_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_variant, num_classes, best_model_path, device, exp_dir=exp_dir)
    loader, class_to_name = _create_loader(model_variant, data_dir, batch_size)

    all_embeddings = []
    all_labels = []
    all_paths = []
    all_species = []

    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            emb = model.get_embeddings(inputs)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            all_paths.extend(paths)
            for lbl in labels:
                all_species.append(class_to_name.get(int(lbl), ""))

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    return embeddings, labels, all_paths, all_species


def extract_logits(
    model_variant: str,
    data_dir: str,
    best_model_path: str,
    num_classes: int,
    batch_size: int = 8,
    device: torch.device | None = None,
    exp_dir: str | None = None,
) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_variant, num_classes, best_model_path, device, exp_dir=exp_dir)
    loader, _ = _create_loader(model_variant, data_dir, batch_size)

    all_logits = []
    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 57,
) -> np.ndarray:
    from sklearn.manifold import TSNE

    n_samples = embeddings.shape[0]
    effective_perp = min(perplexity, max(30, n_samples // 5 - 1))
    logger.info(f"Computing t-SNE {n_components}D (perplexity={effective_perp})...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perp,
        random_state=random_state,
        max_iter=1000,
    )
    return tsne.fit_transform(embeddings)


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 57,
) -> np.ndarray:
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP. Install with: pip install umap-learn")
        return None
    logger.info(f"Computing UMAP {n_components}D...")
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(embeddings)


def save_embeddings(
    output_dir: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    species: list[str],
    class_names: list[str],
    tsne_2d: np.ndarray | None = None,
    tsne_3d: np.ndarray | None = None,
    umap_2d: np.ndarray | None = None,
    umap_3d: np.ndarray | None = None,
):
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    np.save(os.path.join(emb_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(emb_dir, "labels.npy"), labels)
    logger.info(f"Embeddings salvati: {embeddings.shape}")

    if tsne_2d is not None:
        np.save(os.path.join(emb_dir, "tsne_2d.npy"), tsne_2d)
    if tsne_3d is not None:
        np.save(os.path.join(emb_dir, "tsne_3d.npy"), tsne_3d)
    if umap_2d is not None:
        np.save(os.path.join(emb_dir, "umap_2d.npy"), umap_2d)
    if umap_3d is not None:
        np.save(os.path.join(emb_dir, "umap_3d.npy"), umap_3d)

    points = []
    for i in range(len(paths)):
        filename = os.path.basename(paths[i])
        points.append(
            {
                "idx": i,
                "species": species[i] if i < len(species) else "",
                "label": int(labels[i]) if i < len(labels) else -1,
                "filename": filename,
                "image_url": f"/api/dataset/images/train/{species[i]}/{filename}" if species[i] else "",
                "split": "train",
            }
        )

    metadata = {
        "num_points": len(points),
        "num_classes": len(class_names),
        "class_names": class_names,
        "embedding_dim": int(embeddings.shape[1]),
        "points": points,
    }
    meta_path = os.path.join(emb_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata salvato: {meta_path}")


def run_embedding_extraction(
    exp_dir: str,
    model_variant: str,
    num_classes: int,
    class_names: list[str],
    data_dir: str,
    batch_size: int = 8,
    compute_tsne_flag: bool = True,
    compute_umap_flag: bool = True,
):
    results_path = os.path.join(exp_dir, "results.json")
    if not os.path.exists(results_path):
        logger.error(f"results.json non trovato in {exp_dir}. Esegui prima --aggregate-only")
        return

    with open(results_path) as f:
        results = json.load(f)

    best_fold = results.get("best_fold", 0)
    best_model_path = os.path.join(exp_dir, f"best_fold_{best_fold}.pt")
    if not os.path.exists(best_model_path):
        logger.error(f"Best model not found: {best_model_path}")
        return

    logger.info(f"Estrazione embedding da {model_variant} (best fold {best_fold})...")
    embeddings, labels, paths, species = extract_embeddings(
        model_variant=model_variant,
        data_dir=data_dir,
        best_model_path=best_model_path,
        num_classes=num_classes,
        batch_size=batch_size,
        exp_dir=exp_dir,
    )
    logger.info(f"Estratti {embeddings.shape[0]} embedding, dim={embeddings.shape[1]}")

    tsne_2d = compute_tsne(embeddings, n_components=2) if compute_tsne_flag else None
    tsne_3d = compute_tsne(embeddings, n_components=3) if compute_tsne_flag else None
    umap_2d = compute_umap(embeddings, n_components=2) if compute_umap_flag else None
    umap_3d = compute_umap(embeddings, n_components=3) if compute_umap_flag else None

    save_embeddings(
        output_dir=exp_dir,
        embeddings=embeddings,
        labels=labels,
        paths=paths,
        species=species,
        class_names=class_names,
        tsne_2d=tsne_2d,
        tsne_3d=tsne_3d,
        umap_2d=umap_2d,
        umap_3d=umap_3d,
    )
    logger.info("Estrazione embedding completata.")


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
    from app.core.models import create_transform
    from app.preprocessing.dataset_tool import ImageFolderWithName

    model = load_trained_model(model_variant, num_classes, model_path, device, exp_dir=exp_dir)
    model.eval()
    transform = create_transform(model_variant)
    # Se il modello è stato allenato con embeddings L2-normalizzate prima della
    # testa lineare (normalize_embeddings), i logits (Energy detector) devono
    # usare embeddings normalizzate allo stesso modo, altrimenti esplodono.
    normalize_logits = bool(getattr(model, "_config", None) and model._config.normalize_embeddings)

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
                    logit_in = F.normalize(emb, p=2, dim=1) if normalize_logits else emb
                    logit = classifier(logit_in)
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
