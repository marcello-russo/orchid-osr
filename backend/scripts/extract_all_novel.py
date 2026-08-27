"""Extract embeddings + t-SNE/UMAP with novel class (sphegodes_Napoli) for ALL models in exp7, exp8, exp10."""

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
TRAIN_DIR = "dataset/cropped_256x512/train"
TEST_DIR = "dataset/cropped_256x512/test"
BATCH_SIZE = 8

EXPERIMENTS = []
for exp_num, exp_group in [(7, "exp2_supcon"), (8, "exp3_sphor"), (10, "exp10_mgh_ncd")]:
    for model in ["convnext_small", "convnext_tiny", "dinov2_base", "dinov2_small", "resnet18", "resnet50"]:
        if exp_num == 7:
            exp_name = f"exp2_supcon_{model}"
        elif exp_num == 8:
            exp_name = f"exp3_sphor_{model}"
        else:
            exp_name = f"exp10_mgh_ncd_{model}"
        EXPERIMENTS.append((exp_name, model, exp_group))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for exp_name, model_variant, exp_group in EXPERIMENTS:
        exp_dir = f"experiments/{exp_name}"
        results_path = f"{exp_dir}/results.json"
        if not os.path.exists(results_path):
            print(f"SKIP {exp_name}: no results.json")
            continue

        with open(results_path) as f:
            results = json.load(f)

        best_fold = results.get("best_fold", 0)
        best_model_path = f"{exp_dir}/best_fold_{best_fold}.pt"
        if not os.path.exists(best_model_path):
            print(f"SKIP {exp_name}: best_fold_{best_fold}.pt not found")
            continue

        # Load existing training embeddings if available
        old_emb_dir = f"{exp_dir}/embeddings"
        old_meta = f"{old_emb_dir}/metadata.json"
        if os.path.exists(old_meta):
            with open(old_meta) as f:
                metadata = json.load(f)
            old_points = metadata.get("points", [])
            # Check if it already has novel class
            has_novel = any(p.get("split") == "novel" or "sphegodes_Napoli" in p.get("species", "") for p in old_points)
            if has_novel:
                print(f"OK  {exp_name}: already has novel")
                continue

        print(f"GEN  {exp_name} (best fold {best_fold})...")

        # Extract test embeddings (includes novel class)
        test_emb, test_labels, test_paths, test_species = extract_embeddings(
            model_variant=model_variant,
            data_dir=TEST_DIR,
            best_model_path=best_model_path,
            num_classes=len(KNOWN_CLASSES),
            batch_size=BATCH_SIZE,
            exp_dir=exp_dir,
        )
        print(f"  Test: {test_emb.shape[0]} embedding")

        # Load existing training embeddings
        train_emb_file = f"{exp_dir}/embeddings/embeddings.npy"
        if os.path.exists(train_emb_file):
            train_emb = np.load(train_emb_file)
            train_labels_file = f"{exp_dir}/embeddings/labels.npy"
            train_labels = np.load(train_labels_file) if os.path.exists(train_labels_file) else np.zeros(len(train_emb))
            print(f"  Train: {train_emb.shape[0]} embedding")
        else:
            # No existing embeddings - extract from train too
            print("  No train embeddings found, extracting from train set...")
            train_emb, train_labels, _, _ = extract_embeddings(
                model_variant=model_variant,
                data_dir=TRAIN_DIR,
                best_model_path=best_model_path,
                num_classes=len(KNOWN_CLASSES),
                batch_size=BATCH_SIZE,
                exp_dir=exp_dir,
            )

        # Find novel test samples
        novel_mask = np.array([s == NOVEL_CLASS for s in test_species])
        known_test_mask = ~novel_mask

        # Combine
        combined_emb = np.vstack([train_emb, test_emb])
        combined_labels = np.concatenate([train_labels, test_labels])
        n_train = len(train_emb)

        print(
            f"  Combined: {combined_emb.shape} "
            f"(train={n_train}, test_known={known_test_mask.sum()}, novel={novel_mask.sum()})"
        )

        # Compute projections
        tsne_2d = compute_tsne(combined_emb, n_components=2)
        tsne_3d = compute_tsne(combined_emb, n_components=3)
        umap_2d = compute_umap(combined_emb, n_components=2)
        umap_3d = compute_umap(combined_emb, n_components=3)

        # Build points metadata
        class_names = KNOWN_CLASSES + [NOVEL_CLASS]
        points = []
        for i in range(len(combined_emb)):
            if i < n_train:
                species_idx = int(train_labels[i]) if i < len(train_labels) else 0
                species = KNOWN_CLASSES[species_idx] if species_idx < len(KNOWN_CLASSES) else "unknown"
                split = "train"
            else:
                test_i = i - n_train
                species = test_species[test_i]
                split = "novel" if novel_mask[test_i] else "test"
            points.append({"idx": i, "species": species, "filename": "", "split": split})

        metadata = {
            "num_points": len(combined_emb),
            "num_classes": len(class_names),
            "class_names": class_names,
            "embedding_dim": int(combined_emb.shape[1]),
            "points": points,
        }

        out_dir = old_emb_dir
        np.save(os.path.join(out_dir, "embeddings.npy"), combined_emb)
        np.save(os.path.join(out_dir, "labels.npy"), combined_labels)
        np.save(os.path.join(out_dir, "tsne_2d.npy"), tsne_2d)
        np.save(os.path.join(out_dir, "tsne_3d.npy"), tsne_3d)
        np.save(os.path.join(out_dir, "umap_2d.npy"), umap_2d)
        np.save(os.path.join(out_dir, "umap_3d.npy"), umap_3d)
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Salvato in {out_dir}/")

    print("\nDone!")


if __name__ == "__main__":
    main()
