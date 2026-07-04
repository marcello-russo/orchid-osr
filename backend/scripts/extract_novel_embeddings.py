"""Extract embeddings + t-SNE for known + novel (sphegodes_Napoli) classes.

Generates a combined embedding visualization overlay.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.ncd.extract import (
    compute_tsne,
    compute_umap,
    extract_embeddings,
)

KNOWN_CLASSES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. majellensis",
    "O. sphegodes",
    "O. sphegodes_Palena",
]
NOVEL_CLASS = "O. sphegodes_Napoli"

EXP_DIR = "experiments/exp10_mgh_ncd_dinov2_small"
TRAIN_DIR = "dataset/cropped_256x512/train"
TEST_DIR = "dataset/cropped_256x512/test"
OUTPUT_DIR = os.path.join(EXP_DIR, "embeddings_with_novel")
MODEL_VARIANT = "dinov2_small"
BATCH_SIZE = 8


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load results to find best fold
    with open(os.path.join(EXP_DIR, "results.json")) as f:
        results = json.load(f)
    best_fold = results.get("best_fold", 0)
    best_model_path = os.path.join(EXP_DIR, f"best_fold_{best_fold}.pt")
    print(f"Model: {best_model_path} (best fold {best_fold})")

    # Load existing training embeddings
    train_emb = np.load(os.path.join(EXP_DIR, "embeddings", "embeddings.npy"))
    train_labels = np.load(os.path.join(EXP_DIR, "embeddings", "labels.npy"))
    print(f"Train embeddings: {train_emb.shape}, labels: {len(train_labels)}")

    # Extract embeddings from test set (includes novel class)
    print(f"\nExtracting test embeddings from {TEST_DIR}...")
    test_emb, test_labels, test_paths, test_species = extract_embeddings(
        model_variant=MODEL_VARIANT,
        data_dir=TEST_DIR,
        best_model_path=best_model_path,
        num_classes=len(KNOWN_CLASSES),
        batch_size=BATCH_SIZE,
        exp_dir=EXP_DIR,
    )
    print(f"Test embeddings: {test_emb.shape}, labels: {len(test_labels)}")

    # Find which test samples are novel
    novel_mask = np.array([s == NOVEL_CLASS for s in test_species])
    known_test_mask = ~novel_mask
    print(f"Novel (sphegodes_Napoli): {novel_mask.sum()}")
    print(f"Known (test): {known_test_mask.sum()}")

    # Combine train + test embeddings for t-SNE
    combined_emb = np.vstack([train_emb, test_emb])
    combined_labels = np.concatenate([train_labels, test_labels])
    print(f"\nCombined: {combined_emb.shape}")
    print("Computing t-SNE...")
    tsne_2d = compute_tsne(combined_emb, n_components=2)
    tsne_3d = compute_tsne(combined_emb, n_components=3)
    print("Computing UMAP...")
    umap_2d = compute_umap(combined_emb, n_components=2)
    umap_3d = compute_umap(combined_emb, n_components=3)

    # Separate train, known-test, novel indices
    n_train = len(train_emb)
    train_indices = np.arange(n_train)
    known_test_indices = np.where(~novel_mask)[0] + n_train
    novel_indices = np.where(novel_mask)[0] + n_train

    # Save combined metadata
    class_names = KNOWN_CLASSES + [NOVEL_CLASS]
    metadata = {
        "num_points": len(combined_emb),
        "num_classes": len(class_names),
        "class_names": class_names,
        "embedding_dim": combined_emb.shape[1],
        "train_indices": train_indices.tolist(),
        "known_test_indices": known_test_indices.tolist(),
        "novel_indices": novel_indices.tolist(),
    }

    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), combined_emb)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), combined_labels)
    np.save(os.path.join(OUTPUT_DIR, "tsne_2d.npy"), tsne_2d)
    np.save(os.path.join(OUTPUT_DIR, "tsne_3d.npy"), tsne_3d)
    np.save(os.path.join(OUTPUT_DIR, "umap_2d.npy"), umap_2d)
    np.save(os.path.join(OUTPUT_DIR, "umap_3d.npy"), umap_3d)
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSalvato in {OUTPUT_DIR}/")
    print(f"  Train: {len(train_indices)}")
    print(f"  Known test: {len(known_test_indices)}")
    print(f"  Novel: {len(novel_indices)}")


if __name__ == "__main__":
    main()
