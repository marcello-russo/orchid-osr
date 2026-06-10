"""NCD pipeline orchestrator: extraction + clustering + OSR evaluation."""

import logging
import os

import numpy as np

from app.core.ncd.openset import evaluate_osr_novel

logger = logging.getLogger(__name__)


def run_ncd_pipeline(
    exp_dir: str,
    model_variant: str,
    num_classes: int,
    class_names: list[str],
    data_dir: str,
    batch_size: int = 8,
    k_range: list[int] | None = None,
    run_dbscan: bool = True,
    run_spectral: bool = True,
    run_osr: bool = False,
    test_dir: str | None = None,
    exclude_classes: list[str] | None = None,
    osr_method: str = "openmax",
    ncd_mode: bool = False,
    all_classes: list[str] | None = None,
):
    """Full NCD pipeline: extract embeddings → cluster → OSR evaluation.

    Args:
        exp_dir: experiment output directory.
        model_variant: model name.
        num_classes: number of known classes.
        class_names: list of class names.
        data_dir: directory with training images (for embedding extraction).
        batch_size: batch size for extraction.
        k_range: range of k for k-means grid search.
        run_dbscan: whether to run DBSCAN clustering.
        run_spectral: whether to run spectral clustering.
        run_osr: whether to run Open-Set Recognition evaluation.
        test_dir: test directory for OSR evaluation (required if run_osr=True).
    """
    from app.core.ncd.clustering import (
        dbscan_clustering,
        grid_search_kmeans,
        kmeans_clustering,
        spectral_clustering,
    )
    from app.core.ncd.extract import run_embedding_extraction
    from app.core.ncd.metrics import compute_all_metrics

    if k_range is None:
        k_range = list(range(2, 12))

    # Filtra classi escluse (es. hidden novel per NCD)
    if exclude_classes:
        class_names = [c for c in class_names if c not in exclude_classes]

    figures_dir = os.path.join(exp_dir, "ncd_figures")

    # Step 1: extract embeddings
    emb_dir = os.path.join(exp_dir, "embeddings")
    embeddings_path = os.path.join(emb_dir, "embeddings.npy")

    if not os.path.exists(embeddings_path):
        logger.info("Embedding non trovati, estrazione...")
        run_embedding_extraction(
            exp_dir=exp_dir,
            model_variant=model_variant,
            num_classes=num_classes,
            class_names=class_names,
            data_dir=data_dir,
            batch_size=batch_size,
        )
    else:
        logger.info(f"Embedding trovati in {embeddings_path}")

    # Load embeddings
    embeddings = np.load(embeddings_path)
    labels_path = os.path.join(emb_dir, "labels.npy")
    if os.path.exists(labels_path):
        true_labels = np.load(labels_path)
    else:
        logger.error("labels.npy non trovato")
        return

    logger.info(f"Embedding caricati: {embeddings.shape}, labels: {true_labels.shape}")

    # Step 2: cluster + evaluate
    results = {
        "model_variant": model_variant,
        "num_classes": num_classes,
        "class_names": class_names,
        "embedding_shape": list(embeddings.shape),
        "clustering": {},
    }

    # K-means grid search
    logger.info("\n--- K-means grid search ---")
    best_kmeans = grid_search_kmeans(embeddings, true_labels, k_range)
    results["clustering"]["kmeans_best"] = {
        "k": best_kmeans["k"],
        "acc": best_kmeans["acc"],
        "nmi": best_kmeans.get("nmi", 0),
        "ari": best_kmeans.get("ari", 0),
        "ami": best_kmeans.get("ami", 0),
        "f_measure": best_kmeans.get("f1", 0),
        "silhouette": best_kmeans.get("silhouette", 0),
    }
    results["clustering"]["selected_k"] = best_kmeans["k"]
    logger.info(f"K-means best: k={best_kmeans['k']}, ACC={best_kmeans['acc']:.4f}")

    # K-means with true k
    logger.info(f"\n--- K-means (k=true={num_classes}) ---")
    pred_km, info_km = kmeans_clustering(embeddings, num_classes)
    metrics_km = compute_all_metrics(true_labels, pred_km)
    results["clustering"]["kmeans_true_k"] = {**metrics_km, "info": info_km}
    logger.info(f"K-means k={num_classes}: ACC={metrics_km['acc']:.4f}")

    # DBSCAN
    if run_dbscan:
        logger.info("\n--- DBSCAN ---")
        pred_db, info_db = dbscan_clustering(embeddings)
        metrics_db = compute_all_metrics(true_labels, pred_db)
        results["clustering"]["dbscan"] = {**metrics_db, "info": info_db}
        logger.info(f"DBSCAN: ACC={metrics_db['acc']:.4f}, clusters={info_db['n_clusters']}")

    # Spectral
    if run_spectral:
        logger.info(f"\n--- Spectral k={num_classes} ---")
        pred_sp, info_sp = spectral_clustering(embeddings, num_classes)
        metrics_sp = compute_all_metrics(true_labels, pred_sp)
        results["clustering"]["spectral"] = {**metrics_sp, "info": info_sp}
        logger.info(f"Spectral: ACC={metrics_sp['acc']:.4f}")

    # Figures
    if embeddings.shape[1] > 2:
        try:
            from sklearn.manifold import TSNE

            logger.info("Computing t-SNE for figure...")
            tsne = TSNE(n_components=2, random_state=57, perplexity=max(5, min(30, len(embeddings) // 5)))
            emb_2d = tsne.fit_transform(embeddings)
            known_mask = np.ones(len(true_labels), dtype=bool)
            from app.core.ncd.reporting import plot_tsne_known_vs_unknown

            plot_tsne_known_vs_unknown(emb_2d, true_labels, class_names, known_mask, figures_dir)
        except Exception as e:
            logger.warning(f"t-SNE figure failed: {e}")

    # Clustering comparison figure
    try:
        from app.core.ncd.reporting import plot_clustering_comparison

        algo_data = {}
        for algo_name in ["kmeans", "dbscan", "spectral"]:
            if algo_name in results["clustering"]:
                algo_data[algo_name] = results["clustering"][algo_name]
        if algo_data:
            plot_clustering_comparison(algo_data, figures_dir)
    except Exception as e:
        logger.warning(f"Clustering comparison figure failed: {e}")

    # Open-Set Recognition
    if run_osr:
        if test_dir is None:
            logger.warning("test_dir required for OSR, skipping")
        else:
            logger.info(f"\n--- Open-Set Recognition (method={osr_method}) ---")

            if ncd_mode:
                if all_classes is None:
                    logger.warning("all_classes required for ncd_mode, skipping OSR")
                else:
                    import json

                    import torch
                    from torch.utils.data import DataLoader

                    from app.core.models import create_transform
                    from app.core.ncd.extract import load_trained_model
                    from app.preprocessing.dataset_tool import ImageFolderWithName

                    train_classes = class_names
                    novel_classes = [c for c in all_classes if c not in train_classes]
                    if not novel_classes:
                        logger.warning("No novel class found — skipping OSR evaluation")
                    else:
                        novel_name = novel_classes[0]
                        novel_idx = all_classes.index(novel_name)
                        known_idxs = [all_classes.index(c) for c in train_classes]
                        logger.info(f"Evaluating OSR for novel class: {novel_name}")

                        results_path = os.path.join(exp_dir, "results.json")
                        best_fold = 0
                        if os.path.exists(results_path):
                            with open(results_path) as f:
                                best_fold = json.load(f).get("best_fold", 0)
                        best_model_path = os.path.join(exp_dir, f"best_fold_{best_fold}.pt")
                        if not os.path.exists(best_model_path):
                            logger.error(f"Best model not found: {best_model_path}")
                            return

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = load_trained_model(model_variant, num_classes, best_model_path, device)
                        transform = create_transform(model_variant)
                        train_dataset = ImageFolderWithName(data_dir, transform=transform)
                        test_dataset = ImageFolderWithName(test_dir, transform=transform)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                        osr_results = evaluate_osr_novel(
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            novel_label=novel_idx,
                            device=device,
                        )
                        results["osr"] = osr_results
                        logger.info(f"NCD OSR complete. AUROC: {osr_results['auroc']:.4f}")

            elif osr_method == "ensemble":
                import json

                import torch

                from app.core.ncd.extract import extract_embeddings, load_trained_model
                from app.core.ncd.metrics import compute_all_metrics
                from app.core.ncd.openset import (
                    compute_energy_scores,
                    compute_ensemble_scores,
                    compute_knn_scores,
                    compute_mahalanobis_scores,
                    compute_osr_metrics,
                )

                results_path = os.path.join(exp_dir, "results.json")
                best_fold = 0
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        best_fold = json.load(f).get("best_fold", 0)
                best_model_path = os.path.join(exp_dir, f"best_fold_{best_fold}.pt")
                if not os.path.exists(best_model_path):
                    logger.error(f"Best model not found: {best_model_path}")
                    return
                model = load_trained_model(model_variant, num_classes, best_model_path, torch.device("cpu"))
                train_emb, train_labels, _ = extract_embeddings(
                    model_variant, data_dir, best_model_path, num_classes, batch_size, torch.device("cpu")
                )
                test_emb, test_labels, _ = extract_embeddings(
                    model_variant, test_dir, best_model_path, num_classes, batch_size, torch.device("cpu")
                )
                train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-12)
                test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-12)

                maha = compute_mahalanobis_scores(train_norm, train_labels, test_norm, num_classes)
                knn = compute_knn_scores(train_norm, test_norm)
                energy = compute_energy_scores(test_norm)
                ensemble = compute_ensemble_scores(maha, knn, energy)

                unknown_labels = np.full(len(test_labels), -1)
                all_metrics = []
                for left_out in range(num_classes):
                    val_labels = unknown_labels.copy()
                    val_labels[test_labels != left_out] = test_labels[test_labels != left_out]
                    val_known = [c for c in range(num_classes) if c != left_out]
                    metrics = compute_osr_metrics(ensemble, val_labels, val_known)
                    metrics["left_out"] = class_names[left_out]
                    all_metrics.append(metrics)

                osr_results = {
                    "model_variant": model_variant,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "mean_auroc": float(np.mean([m["auroc"] for m in all_metrics])),
                    "per_class_metrics": all_metrics,
                    "method": "ensemble",
                    "ensemble_scores": {
                        "mahalanobis": float(
                            np.mean(
                                [
                                    compute_osr_metrics(
                                        maha,
                                        np.full(len(test_labels), -1) if _ else test_labels,
                                        list(range(num_classes)),
                                    )["auroc"]
                                    for _ in [0]
                                ]
                            )
                        ),
                    },
                }
                logger.info(f"Ensemble OSR complete. Mean AUROC: {osr_results['mean_auroc']:.4f}")
            else:
                from app.core.ncd.openset import run_openset_evaluation

                osr_results = run_openset_evaluation(
                    exp_dir=exp_dir,
                    model_variant=model_variant,
                    num_classes=num_classes,
                    class_names=class_names,
                    train_dir=data_dir,
                    test_dir=test_dir,
                    batch_size=batch_size,
                )
            results["openset"] = osr_results

            # OSR figures
            try:
                from app.core.ncd.reporting import plot_auroc_per_class, plot_oscr_curve

                per_class = osr_results.get("per_class_metrics", [])
                if per_class:
                    plot_auroc_per_class(per_class, osr_results.get("mean_auroc", 0.0), figures_dir)
                oscr = osr_results.get("oscr_curve", [])
                baseline_oscr = osr_results.get("baseline_oscr_curve", [])
                if oscr:
                    plot_oscr_curve(oscr, baseline_oscr or None, model_variant, figures_dir)
            except Exception as e:
                logger.warning(f"OSR figures failed: {e}")

    # Save results
    ncd_path = os.path.join(exp_dir, "ncd_results.json")
    with open(ncd_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nRisultati NCD salvati in: {ncd_path}")

    return results
