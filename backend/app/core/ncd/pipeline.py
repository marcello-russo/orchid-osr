"""NCD pipeline orchestrator: extraction -> clustering -> multi-detector OSR."""

from __future__ import annotations

import json
import logging
import os
import threading

import numpy as np
import torch

from app.core.ncd.detector import get_detector, list_detectors
from app.core.ncd.evaluation import evaluate_loo, evaluate_novel, evaluate_osr
from app.core.ncd.extract import extract_all

logger = logging.getLogger(__name__)


def _get_device(timeout: float = 5.0) -> torch.device:
    """Safe device detection with timeout.

    torch.cuda.is_available() puo' bloccare se il driver NVIDIA e' in stato
    inconsistente (es. dopo sospensione del portatile). Usa un thread separato
    con timeout per evitare freeze della pipeline.
    Se NCD_FORCE_CPU=1, salta completamente la verifica CUDA.
    """
    if os.environ.get("NCD_FORCE_CPU") == "1":
        return torch.device("cpu")

    result = [False]

    def check():
        try:
            result[0] = torch.cuda.is_available()
        except Exception:
            result[0] = False

    t = threading.Thread(target=check, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logger.warning(
            "torch.cuda.is_available() non risponde dopo %.1fs — fallback a CPU. "
            "Imposta NCD_FORCE_CPU=1 per saltare la verifica CUDA.",
            timeout,
        )
        return torch.device("cpu")

    return torch.device("cuda" if result[0] else "cpu")


def drop_hidden_classes(bundle, exclude_classes):
    """Toglie le classi nascoste dal train e RINUMERA le label.

    La GCD costruisce lo spazio delle label da `train_class_names` e usa
    `label < num_classes` come maschera dei campioni etichettati. Lasciando gli
    indici originali (con il buco della classe nascosta) la maschera scambia
    noti e novel: e' successo davvero, con ACC new 0.996 su una specie che il
    modello non aveva mai visto. Il train dir contiene tutte le specie, quindi
    il filtro deve stare qui e non in symlink creati a mano.
    """
    if not exclude_classes or not bundle.train_class_names:
        return bundle
    kept_classes = [c for c in bundle.train_class_names if c not in exclude_classes]
    old_names = list(bundle.train_class_names)
    keep = np.array([old_names[int(label)] in kept_classes for label in bundle.train_labels])
    bundle.train_embeddings = bundle.train_embeddings[keep]
    bundle.train_labels = np.array(
        [kept_classes.index(old_names[int(label)]) for label in bundle.train_labels[keep]],
        dtype=np.int64,
    )
    bundle.train_class_names = kept_classes
    logger.info(
        "Filtrate %d immagini delle classi nascoste dal train (restano %d, classi %s)",
        int((~keep).sum()),
        int(keep.sum()),
        kept_classes,
    )
    return bundle


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

    device = _get_device()

    collect_logits = any(get_detector(n).requires_logits for n in list_detectors())

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

    # Le classi nascoste non devono entrare nel fit dei rivelatori né nel
    # labeled set della GCD. Il train dir le contiene tutte, e la pipeline si
    # affidava a symlink creati a mano (`train_ncd_exp4`/`train_ncd_exp5`) che
    # non esistono nel dataset HF: senza questo filtro l'AUROC cade a 0.5 e la
    # SS-KM "scopre" una specie che era già nel train (ACC new 0.996).
    bundle = drop_hidden_classes(bundle, exclude_classes)

    results: dict = {
        "model_variant": model_variant,
        "num_classes": num_classes,
        "class_names": class_names,
        "embedding_shape": list(bundle.train_embeddings.shape),
        "clustering": {},
    }

    if ncd_mode and test_dir and os.path.exists(test_dir):
        # Build consistent label space: train labels are 0..K-1,
        # test labels remapped to match train for known classes,
        # novel classes get new indices (K, K+1, ...)
        all_names = list(bundle.train_class_names)
        for c in bundle.test_class_names:
            if c not in all_names:
                all_names.append(c)
        name_to_label = {c: i for i, c in enumerate(all_names)}

        # Remap test labels to consistent label space
        remapped_test_labels = np.array([name_to_label[bundle.test_class_names[label]] for label in bundle.test_labels])

        combined_emb = np.vstack([bundle.train_embeddings, bundle.test_embeddings])
        combined_labels = np.hstack([bundle.train_labels, remapped_test_labels])
        combined_n_classes = len(all_names)
        logger.info(
            "Combined labels: %d classes (%s), remapped test labels for %d train + %d novel",
            combined_n_classes,
            all_names,
            len(bundle.train_class_names),
            combined_n_classes - len(bundle.train_class_names),
        )
    else:
        combined_emb = bundle.train_embeddings
        combined_labels = bundle.train_labels
        combined_n_classes = num_classes

    from app.core.ncd.clustering import (
        dbscan_clustering,
        grid_search_kmeans,
        kmeans_clustering,
        spectral_clustering,
        ss_kmeans_clustering,
    )
    from app.core.ncd.evaluation import compute_all_metrics, compute_ari, compute_nmi, gcd_old_new_all_acc

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

    true_k = combined_n_classes if ncd_mode else num_classes
    logger.info("\n--- K-means (k=true=%d) ---", true_k)
    pred_km, info_km = kmeans_clustering(combined_emb, true_k)
    metrics_km = compute_all_metrics(combined_labels, pred_km)
    results["clustering"]["kmeans_true_k"] = {**metrics_km, "info": info_km}
    logger.info("K-means k=%d: ACC=%.4f", true_k, metrics_km["acc"])

    if run_dbscan:
        logger.info("\n--- DBSCAN ---")
        pred_db, info_db = dbscan_clustering(combined_emb)
        metrics_db = compute_all_metrics(combined_labels, pred_db)
        results["clustering"]["dbscan"] = {**metrics_db, "info": info_db}

    if run_spectral:
        logger.info("\n--- Spectral k=%d ---", true_k)
        pred_sp, info_sp = spectral_clustering(combined_emb, true_k)
        metrics_sp = compute_all_metrics(combined_labels, pred_sp)
        results["clustering"]["spectral"] = {**metrics_sp, "info": info_sp}

    # ---- GCD (Vaze et al. 2022): SS-KMeans vincolato + ACC Old/New/All ----
    # In ncd_mode il pool combinato contiene classi note (train+test) e novel:
    # labeled = campioni train, pool = combined (protocollo Generalized Category Discovery).
    if ncd_mode and all_classes:
        novel_candidates = [c for c in all_classes if c not in set(bundle.train_class_names)]
        if novel_candidates and not run_osr:
            results["novel_classes"] = novel_candidates
        try:
            labeled_mask = combined_labels < num_classes
            pool_norm = combined_emb / (np.linalg.norm(combined_emb, axis=1, keepdims=True) + 1e-12)
            labeled_lab = combined_labels[labeled_mask]
            n_known = num_classes
            n_novel = combined_n_classes - n_known

            pred_ss, _ = ss_kmeans_clustering(
                pool_norm,
                labeled_mask,
                labeled_lab,
                n_known=n_known,
                n_novel=n_novel,
            )
            gcd_ss = gcd_old_new_all_acc(combined_labels, pred_ss, labeled_mask)
            results["gcd"] = {
                "protocol": "GCD (Vaze et al. 2022)",
                "novel_classes": novel_candidates,
                "ss_kmeans": {
                    **gcd_ss,
                    "nmi": compute_nmi(combined_labels, pred_ss),
                    "ari": compute_ari(combined_labels, pred_ss),
                },
            }
            logger.info(
                "\n=== GCD (SS-KMeans) === All=%.4f Old=%.4f New=%.4f NMI=%.4f",
                gcd_ss["acc_all"],
                gcd_ss["acc_old"],
                gcd_ss["acc_new"],
                results["gcd"]["ss_kmeans"]["nmi"],
            )
        except Exception as e:
            logger.warning("GCD SS-KMeans fallito: %s", e)
            import traceback

            traceback.print_exc()

    if run_osr:
        novel_candidates: list[str] = []
        if ncd_mode and all_classes:
            # dalle label effettivamente presenti, non dall'elenco delle
            # cartelle: dopo il filtro sopra le due cose non coincidono più
            train_classes = {bundle.train_class_names[int(i)] for i in np.unique(bundle.train_labels)}
            novel_candidates = [c for c in all_classes if c not in train_classes]
            if novel_candidates:
                logger.info("Novel class(es): %s", novel_candidates)
            else:
                logger.warning("Nessuna novel class in all_classes vs train")

        results["detectors"] = {}
        if novel_candidates:
            # per compatibilità singola novel + supporto 2 novel
            results["novel_class"] = novel_candidates[0] if len(novel_candidates) == 1 else None
            results["novel_classes"] = novel_candidates

        osr_scores: dict[str, np.ndarray] = {}
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
                osr_scores[det_name] = np.asarray(scores, dtype=float)

                if ncd_mode and novel_candidates:
                    if len(novel_candidates) == 1:
                        results["detectors"][det_name] = evaluate_novel(
                            scores,
                            bundle.test_labels,
                            novel_candidates[0],
                            bundle.test_class_names,
                        )
                    else:
                        # 2 novel: 3 AUROC (globale + per-novel)
                        results["detectors"][det_name] = evaluate_osr(
                            scores,
                            bundle.test_labels,
                            bundle.test_class_names,
                            novel_candidates,
                        )
                else:
                    results["detectors"][det_name] = evaluate_loo(scores, bundle.test_labels, bundle.test_class_names)
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
        if ncd_mode and novel_candidates:
            try:
                novel_idx = [bundle.test_class_names.index(c) for c in novel_candidates]
                y_true_osr = np.isin(np.asarray(bundle.test_labels), novel_idx).astype(np.int64)
                np.savez(
                    os.path.join(exp_dir, "osr_scores.npz"),
                    y_true=y_true_osr,
                    class_names=np.asarray(bundle.test_class_names),
                    **osr_scores,
                )
                logger.info("OSR scores salvati in: %s", os.path.join(exp_dir, "osr_scores.npz"))
            except Exception as e:
                logger.warning("Salvataggio osr_scores.npz fallito (ncd_results.json intatto): %s", e)

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
            known_mask = combined_labels < num_classes if ncd_mode else np.ones(len(combined_labels), dtype=bool)
            if ncd_mode:
                combined_class_names = list(bundle.train_class_names)
                if novel_candidates:
                    combined_class_names.extend(novel_candidates)
            else:
                combined_class_names = class_names
            plot_tsne_known_vs_unknown(
                emb_2d,
                combined_labels,
                combined_class_names,
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
