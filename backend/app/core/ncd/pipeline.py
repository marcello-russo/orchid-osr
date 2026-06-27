"""NCD pipeline orchestrator: extraction -> clustering -> multi-detector OSR."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

from app.core.ncd.detector import get_detector, list_detectors
from app.core.ncd.evaluation import evaluate_loo, evaluate_novel
from app.core.ncd.extraction import extract_all

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
    if k_range is None:
        k_range = list(range(2, 12))

    if ncd_mode and all_classes is None:
        logger.error("ncd_mode=True richiede all_classes nel chiamante")
        return
    if run_osr and test_dir is None:
        logger.error("run_osr=True richiede test_dir")
        return

    if exclude_classes:
        class_names = [c for c in class_names if c not in exclude_classes]

    figures_dir = os.path.join(exp_dir, "ncd_figures")
    os.makedirs(figures_dir, exist_ok=True)

    results_path = os.path.join(exp_dir, "results.json")
    best_fold = 0
    if os.path.exists(results_path):
        with open(results_path) as f:
            best_fold = json.load(f).get("best_fold", 0)
    best_model_path = os.path.join(exp_dir, f"best_fold_{best_fold}.pt")

    if not os.path.exists(best_model_path):
        logger.error("Best model not found: %s", best_model_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    collect_logits = any(
        get_detector(n).requires_logits for n in list_detectors()
    )

    bundle = extract_all(
        model_variant=model_variant,
        train_dir=data_dir,
        test_dir=test_dir or data_dir,
        model_path=best_model_path,
        num_classes=num_classes,
        batch_size=batch_size,
        device=device,
        collect_logits=collect_logits,
        exp_dir=exp_dir,
    )

    results: dict = {
        "model_variant": model_variant,
        "num_classes": num_classes,
        "class_names": class_names,
        "embedding_shape": list(bundle.train_embeddings.shape),
        "clustering": {},
    }

    if ncd_mode and test_dir and os.path.exists(test_dir):
        combined_emb = np.vstack([bundle.train_embeddings, bundle.test_embeddings])
        combined_labels = np.hstack([bundle.train_labels, bundle.test_labels])
    else:
        combined_emb = bundle.train_embeddings
        combined_labels = bundle.train_labels

    from app.core.ncd.clustering import (
        dbscan_clustering,
        grid_search_kmeans,
        kmeans_clustering,
        spectral_clustering,
    )
    from app.core.ncd.metrics import compute_all_metrics

    logger.info("\n--- K-means grid search ---")
    best_kmeans = grid_search_kmeans(combined_emb, combined_labels, k_range)
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

    logger.info("\n--- K-means (k=true=%d) ---", num_classes)
    pred_km, info_km = kmeans_clustering(combined_emb, num_classes)
    metrics_km = compute_all_metrics(combined_labels, pred_km)
    results["clustering"]["kmeans_true_k"] = {**metrics_km, "info": info_km}
    logger.info("K-means k=%d: ACC=%.4f", num_classes, metrics_km["acc"])

    if run_dbscan:
        logger.info("\n--- DBSCAN ---")
        pred_db, info_db = dbscan_clustering(combined_emb)
        metrics_db = compute_all_metrics(combined_labels, pred_db)
        results["clustering"]["dbscan"] = {**metrics_db, "info": info_db}

    if run_spectral:
        logger.info("\n--- Spectral k=%d ---", num_classes)
        pred_sp, info_sp = spectral_clustering(combined_emb, num_classes)
        metrics_sp = compute_all_metrics(combined_labels, pred_sp)
        results["clustering"]["spectral"] = {**metrics_sp, "info": info_sp}

    if run_osr:
        novel_name: str | None = None
        if ncd_mode and all_classes:
            train_classes = set(bundle.train_class_names)
            novel_candidates = [c for c in all_classes if c not in train_classes]
            if novel_candidates:
                novel_name = novel_candidates[0]
                logger.info("Novel class: %s", novel_name)
            else:
                logger.warning("Nessuna novel class in all_classes vs train")

        results["detectors"] = {}

        for det_name in list_detectors():
            detector = get_detector(det_name)

            if detector.requires_logits and not collect_logits:
                logger.warning("Skip %s: logits non collezionati", det_name)
                continue

            try:
                detector.fit(bundle.train_embeddings, bundle.train_labels)
                scores = detector.score(
                    bundle.test_embeddings,
                    logits=bundle.test_logits if detector.requires_logits else None,
                )

                if ncd_mode and novel_name:
                    results["detectors"][det_name] = evaluate_novel(
                        scores,
                        bundle.test_labels,
                        novel_name,
                        bundle.test_class_names,
                    )
                else:
                    results["detectors"][det_name] = evaluate_loo(
                        scores, bundle.test_labels, bundle.test_class_names
                    )
            except Exception as e:
                logger.warning("Detector %s fallito: %s", det_name, e)
                import traceback

                traceback.print_exc()
                results["detectors"][det_name] = {"error": str(e)}

        logger.info("\n=== OSR Results ===")
        for det_name, metrics in results["detectors"].items():
            if "error" in metrics:
                logger.info("  %20s: ERROR — %s", det_name, metrics["error"])
            else:
                au = metrics.get("auroc", metrics.get("mean_auroc", float("nan")))
                logger.info("  %20s: AUROC = %.4f", det_name, au)

    if combined_emb.shape[1] > 2:
        try:
            from sklearn.manifold import TSNE
            from app.core.ncd.reporting import plot_tsne_known_vs_unknown

            tsne = TSNE(
                n_components=2,
                random_state=57,
                perplexity=max(5, min(30, len(combined_emb) // 5)),
            )
            emb_2d = tsne.fit_transform(combined_emb)
            known_mask = (
                combined_labels < num_classes
                if ncd_mode
                else np.ones(len(combined_labels), dtype=bool)
            )
            plot_tsne_known_vs_unknown(
                emb_2d,
                combined_labels,
                all_classes if ncd_mode and all_classes else class_names,
                known_mask,
                figures_dir,
            )
        except Exception as e:
            logger.warning("t-SNE figure fallita: %s", e)

    ncd_path = os.path.join(exp_dir, "ncd_results.json")
    with open(ncd_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Risultati NCD salvati in: %s", ncd_path)

    return results
