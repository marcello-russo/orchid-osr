#!/usr/bin/env python3
"""Rigenera gli embedding del viewer per Exp 4 con il setup OSR corretto.

Estrae embeddings dal test_ncd_exp4 (5 classi note + O. garganica novel,
300 punti) con il modello del best fold, e genera il metadata con la
novel class marcata separatamente. Sovrascrive experiments/exp4_sphor_ncd_*/embeddings/.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from torch.utils.data import DataLoader  # noqa: E402

from app.core.models import create_transform  # noqa: E402
from app.preprocessing.dataset_tool import ImageFolderWithName  # noqa: E402

NOVEL = "O. garganica"
MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]


def extract_exp9(model_variant: str, device: torch.device):
    from app.core.ncd.extract import load_trained_model

    exp_dir = BACKEND_DIR / "experiments" / f"exp4_sphor_ncd_{model_variant}"
    results = json.load(open(exp_dir / "results.json"))
    best_fold = results.get("best_fold")
    model_path = exp_dir / f"best_fold_{best_fold}.pt"
    num_classes = 5  # 5 classi note nel training exp9

    model = load_trained_model(model_variant, num_classes, str(model_path), device, exp_dir=str(exp_dir))
    model.eval()
    transform = create_transform(model_variant)

    # Estrai dal test_ncd_exp4 (5 note + novel garganica)
    test_dir = BACKEND_DIR / "dataset" / "cropped_224x224" / "test_ncd_exp4"
    dataset = ImageFolderWithName(str(test_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    embs, labels, paths = [], [], []
    with torch.no_grad():
        for inputs, lbls, pths in loader:
            inputs = inputs.to(device, non_blocking=True)
            emb = model.get_embeddings(inputs).cpu().numpy()
            embs.append(emb)
            labels.append(lbls.numpy())
            paths.extend(pths)

    embeddings = np.concatenate(embs)
    labels = np.concatenate(labels)

    # class_names reali del dataset test (ordine ImageFolder)
    test_classes = dataset.classes  # 6 classi: 5 note + garganica
    # rimappa: le 5 note = 0..4, garganica novel = 5
    name_to_label = {c: i for i, c in enumerate(test_classes)}
    # nota: l'ordine del dataset mette garganica da qualche parte; mappa esplicita
    known = [c for c in test_classes if c != NOVEL]
    known.sort()
    ordered = known + [NOVEL]  # 0..4 note, 5 novel
    label_map = {name_to_label[c]: i for i, c in enumerate(ordered)}

    mapped = np.array([label_map[int(lbl)] for lbl in labels])

    # t-SNE / UMAP
    tsne_2d = _tsne(embeddings, 2)
    tsne_3d = _tsne(embeddings, 3)
    umap_2d = _umap(embeddings, 2)
    umap_3d = _umap(embeddings, 3)

    # salva
    save_dir = exp_dir / "embeddings"
    save_dir.mkdir(exist_ok=True)
    np.save(save_dir / "embeddings.npy", embeddings)
    np.save(save_dir / "labels.npy", mapped)
    np.save(save_dir / "tsne_2d.npy", tsne_2d)
    np.save(save_dir / "tsne_3d.npy", tsne_3d)
    np.save(save_dir / "umap_2d.npy", umap_2d)
    np.save(save_dir / "umap_3d.npy", umap_3d)

    points = []
    for i in range(len(paths)):
        fn = os.path.basename(paths[i])
        sp = ordered[int(mapped[i])]
        is_novel = sp == NOVEL
        points.append(
            {
                "idx": i,
                "species": sp,
                "label": int(mapped[i]),
                "filename": fn,
                "image_url": f"/api/dataset/images/test_ncd_exp4/{sp}/{fn}",
                "split": "novel" if is_novel else "known",
            }
        )
    metadata = {
        "num_points": len(points),
        "num_classes": len(ordered),
        "class_names": ordered,
        "embedding_dim": int(embeddings.shape[1]),
        "novel_class": NOVEL,
        "points": points,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    n_novel = sum(1 for p in points if p["split"] == "novel")
    print(
        f"  {model_variant}: {len(points)} punti (5 note + {n_novel} novel), " f"dim={embeddings.shape[1]}", flush=True
    )


def _tsne(emb, n):
    from sklearn.manifold import TSNE

    return TSNE(n_components=n, random_state=57, perplexity=min(30, max(5, len(emb) // 5))).fit_transform(emb)


def _umap(emb, n):
    try:
        import umap

        return umap.UMAP(n_components=n, random_state=57).fit_transform(emb)
    except Exception:
        # fallback a PCA se umap non disponibile
        from sklearn.decomposition import PCA

        return PCA(n_components=n, random_state=57).fit_transform(emb)


def main():
    if not torch.cuda.is_available():
        print("CUDA richiesta. Abort.", flush=True)
        sys.exit(1)
    device = torch.device("cuda")
    for m in MODELS:
        print(f"Extract exp9_{m}...", flush=True)
        try:
            extract_exp9(m, device)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  FAIL {m}: {e}", flush=True)
    print("=== FATTO ===", flush=True)


if __name__ == "__main__":
    main()
