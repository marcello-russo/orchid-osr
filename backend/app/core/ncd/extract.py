"""Embedding extraction + t-SNE/UMAP dimensionality reduction."""

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_trained_model(
    model_variant: str,
    num_classes: int,
    best_model_path: str,
    device: torch.device,
):
    """Load a trained model from best fold weights."""
    from app.core.models import create_model

    model = create_model(model_variant, num_classes=num_classes, pretrained=False)
    model.load_weights(best_model_path)
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings(
    model_variant: str,
    data_dir: str,
    best_model_path: str,
    num_classes: int,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_variant, num_classes, best_model_path, device)
    from app.core.models import create_transform
    from app.preprocessing.dataset_tool import ImageFolderWithName

    transform = create_transform(model_variant)
    dataset = ImageFolderWithName(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_to_name = {v: k for k, v in dataset.class_to_idx.items()}

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
) -> np.ndarray:
    """Extract classification logits (pre-softmax) from a trained model.

    Unlike extract_embeddings which bypasses the classifier head, this
    runs the full forward pass to get logits for energy-based OOD scoring.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_variant, num_classes, best_model_path, device)
    from app.core.models import create_transform
    from app.preprocessing.dataset_tool import ImageFolderWithName

    transform = create_transform(model_variant)
    dataset = ImageFolderWithName(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
