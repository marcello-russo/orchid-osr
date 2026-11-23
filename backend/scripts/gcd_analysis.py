"""Generalized Category Discovery (GCD) analysis — Vaze et al., CVPR 2022.

Protocollo:
  Exp 4: 5 note (garganica nascosta), test 300 = 250 old + 50 new
  Exp 5: 4 note (garganica+majellensis nascoste), test 300 = 200 old + 100 new
  - labeled set = train embeddings delle classi note
  - pool non etichettato = test set intero (old + novel, K note+novel)
  - SS-KMeans vincolato: centroidi note da medie labeled, novel farthest-first
  - baseline: K-means non informato (10 seed)
  - metriche: ACC Old/New/All (Hungarian), per-novel, NMI, ARI

Uso:
  python scripts/gcd_analysis.py --exp 4                  # Exp 4 (1 novel)
  python scripts/gcd_analysis.py --exp 5                  # Exp 5 (2 novel)
  python scripts/gcd_analysis.py --exp 5 dinov2_small     # un solo modello
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cli.definitions import resolve_experiment
from app.core.ncd.clustering import kmeans_clustering, ss_kmeans_clustering
from app.core.ncd.evaluation import compute_ari, compute_nmi, gcd_old_new_all_acc, per_novel_acc
from app.core.ncd.extract import extract_embeddings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]

BATCH_SIZE = {
    "resnet18": 64,
    "resnet50": 48,
    "convnext_tiny": 48,
    "convnext_small": 32,
    "dinov2_small": 32,
    "dinov2_base": 16,
}

ALL_SPECIES = sorted(
    ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]
)


def _get_device(timeout: float = 5.0):
    import torch

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
        logger.warning("CUDA check timeout — fallback CPU")
        return torch.device("cpu")
    import torch

    return torch.device("cuda" if result[0] else "cpu")


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _extract_pool(exp_dir: str, model_variant: str, config: dict, device, test: bool):
    """Estrae (con cache) embedding di train o test."""
    if test:
        cache = os.path.join(exp_dir, "embeddings", "test_embeddings_full.npy")
        data_dir = config.get("test_dataset_dir") or config["raw_dataset_dir"].replace("/train", "/test")
    else:
        cache = os.path.join(exp_dir, "embeddings", "train_embeddings_known.npy")
        data_dir = config["raw_dataset_dir"]
        # per exp5 (2 novel) il raw contiene tutte le 6 specie: filtriamo dopo
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    cache_sp = cache.replace(".npy", "_species.json")
    if os.path.exists(cache) and os.path.exists(cache_sp):
        logger.info("  cache hit: %s", cache)
        return np.load(cache), json.load(open(cache_sp))

    logger.info("  estrazione %s da %s ...", "test" if test else "train_known", data_dir)
    emb, _, _, species = extract_embeddings(
        model_variant=model_variant,
        data_dir=data_dir,
        best_model_path=os.path.join(
            exp_dir, f"best_fold_{json.load(open(os.path.join(exp_dir,'results.json')))['best_fold']}.pt"
        ),
        num_classes=config["num_classes"],
        batch_size=BATCH_SIZE.get(model_variant, 16),
        device=device,
        exp_dir=exp_dir,
    )
    if not test:
        # filtra le novel dal train
        cfg_novel = list(
            resolve_experiment(int(os.path.basename(exp_dir).split("_")[0].replace("exp", ""))).get(
                "exclude_classes", []
            )
        )
        # fallback: usa exclude_classes del config caricato
        novel_set = set(cfg_novel) or set(config.get("exclude_classes", []))
        keep = [i for i, s in enumerate(species) if s not in novel_set]
        emb = emb[keep]
        species = [species[i] for i in keep]
    np.save(cache, emb)
    json.dump(species, open(cache_sp, "w"))
    return emb, species


def analyze_model(exp_dir: str, device, exp_id: int) -> dict | None:
    model_variant = os.path.basename(exp_dir).replace(f"exp{exp_id}_sphor_ncd_", "")
    config = json.load(open(os.path.join(exp_dir, "config.json")))
    novel_names = list(resolve_experiment(exp_id)["exclude_classes"])
    n_novel = len(novel_names)
    known_classes = [c for c in ALL_SPECIES if c not in novel_names]
    n_known = len(known_classes)

    # ---- Test pool (300): estrazione propria con cache ----
    test_emb_raw, test_species = _extract_pool(exp_dir, model_variant, config, device, test=True)
    test_emb = l2norm(test_emb_raw)
    known_to_idx = {c: i for i, c in enumerate(known_classes)}
    test_true = np.array(
        [known_to_idx[s] if s in known_to_idx else n_known + novel_names.index(s) for s in test_species]
    )
    test_species_arr = np.array(test_species)
    test_known_mask = np.array([s in known_to_idx for s in test_species])

    # ---- Train known (labeled): 1.629 (exp4) o 1.477 (exp5) ----
    train_emb_raw, train_species = _extract_pool(exp_dir, model_variant, config, device, test=False)
    train_emb = l2norm(train_emb_raw)
    train_labels = np.array([known_to_idx[s] for s in train_species])

    # sanity sul test pool
    if exp_id == 4:
        assert (
            len(test_species) == 300 and test_known_mask.sum() == 250
        ), f"exp4 pool atteso 250/50, trovato {test_known_mask.sum()}/{len(test_species)-test_known_mask.sum()}"
    elif exp_id == 5:
        assert len(test_species) == 300 and (test_species_arr == "O. garganica").sum() == 50, "exp5 GAR 50 attesi"
        assert (test_species_arr == "O. majellensis").sum() == 50, "exp5 MAJ 50 attesi"

    # ---- Protocollo GCD ----
    pool_emb = np.vstack([train_emb, test_emb])
    labeled_mask = np.concatenate([np.ones(len(train_emb), bool), np.zeros(len(test_emb), bool)])
    pool_true = np.concatenate([train_labels, test_true])
    np.concatenate([np.array(train_species), test_species_arr])

    out: dict = {
        "protocol": f"GCD (Vaze et al. 2022) — Exp {exp_id}, {n_known} note + {n_novel} novel",
        "novel_classes": novel_names,
        "n_labeled": int(len(train_emb)),
        "n_unlabeled": int(len(test_emb)),
        "n_pool": int(len(pool_emb)),
        "n_old_test": int(test_known_mask.sum()),
        "n_new_test": int((~test_known_mask).sum()),
        "l2_normalized": True,
    }

    # SS-KMeans vincolato
    pred_ss, info_ss = ss_kmeans_clustering(pool_emb, labeled_mask, train_labels, n_known=n_known, n_novel=n_novel)
    from app.core.ncd.evaluation import hungarian_mapping

    mapping = hungarian_mapping(pool_true, pred_ss)
    idx_test = np.arange(len(train_emb), len(pool_emb))
    mapped_test = np.array([mapping.get(int(p), -1) for p in pred_ss[idx_test]])
    correct_test = mapped_test == pool_true[idx_test]
    # per-novel breakdown (test only)
    mask_new_test = ~test_known_mask
    per_novel_ss = {}
    for sp in novel_names:
        m = test_species_arr == sp
        per_novel_ss[sp] = float(correct_test[m].mean()) if m.sum() else float("nan")

    out["ss_kmeans"] = {
        "acc_all_test": float(correct_test.mean()),
        "acc_old_test": float(correct_test[test_known_mask].mean()) if test_known_mask.sum() else float("nan"),
        "acc_new_test": float(correct_test[mask_new_test].mean()) if mask_new_test.sum() else float("nan"),
        "acc_all_pool": float((np.array([mapping.get(int(p), -1) for p in pred_ss]) == pool_true).mean()),
        "per_novel_test": per_novel_ss,
        "nmi_pool": compute_nmi(pool_true, pred_ss),
        "ari_pool": compute_ari(pool_true, pred_ss),
        "n_iter": info_ss["n_iter"],
        "cluster_sizes": info_ss["cluster_sizes"],
    }

    # Baseline K-means (solo test pool, 10 seed)
    accs: dict[str, list[float]] = {"acc_all": [], "acc_old": [], "acc_new": [], "nmi": [], "ari": []}
    per_novel_km: dict[str, list[float]] = {sp: [] for sp in novel_names}
    for seed in range(57, 67):
        pred_km, _ = kmeans_clustering(test_emb, n_clusters=n_known + n_novel, random_state=seed)
        gcd_km = gcd_old_new_all_acc(test_true, pred_km, test_known_mask)
        for k in ("acc_all", "acc_old", "acc_new"):
            accs[k].append(gcd_km[k])
        accs["nmi"].append(compute_nmi(test_true, pred_km))
        accs["ari"].append(compute_ari(test_true, pred_km))
        per_km = per_novel_acc(test_true, pred_km, test_species_arr, novel_names)
        for sp in novel_names:
            per_novel_km[sp].append(per_km[sp])

    out["kmeans"] = {
        "n_seeds": len(accs["acc_all"]),
        **{f"{k}_mean": float(np.mean(v)) for k, v in accs.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in accs.items()},
        "per_novel_mean": {sp: float(np.mean(v)) for sp, v in per_novel_km.items()},
        "per_novel_std": {sp: float(np.std(v)) for sp, v in per_novel_km.items()},
    }

    # OSR multi-novel (detector senza logit, su train->test)
    from app.core.ncd.detector import get_detector
    from app.core.ncd.evaluation import evaluate_osr

    CLASS_NAMES_6 = ALL_SPECIES
    test_loo = np.array([CLASS_NAMES_6.index(s) for s in test_species])  # 0..5 nello spazio 6 classi
    out["osr"] = {}
    for det_name in ["Mahalanobis", "KNN"]:
        det = get_detector(det_name)
        det.fit(train_emb, train_labels)
        scores = det.score(test_emb, logits=None)
        out["osr"][det_name] = evaluate_osr(scores, test_loo, CLASS_NAMES_6, novel_names)

    out_path = os.path.join(exp_dir, "gcd_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(
        "[%s] Exp%d SS-KM(test): All=%.3f Old=%.3f New=%.3f | KMeans: All=%.3f Old=%.3f New=%.3f",
        model_variant,
        exp_id,
        out["ss_kmeans"]["acc_all_test"],
        out["ss_kmeans"]["acc_old_test"],
        out["ss_kmeans"]["acc_new_test"],
        out["kmeans"]["acc_all_mean"],
        out["kmeans"]["acc_old_mean"],
        out["kmeans"]["acc_new_mean"],
    )
    if n_novel == 2:
        logger.info(
            "  per-novel SS-KM: %s | KMeans: %s",
            {k: round(v, 3) for k, v in out["ss_kmeans"]["per_novel_test"].items()},
            {k: round(v, 3) for k, v in out["kmeans"]["per_novel_mean"].items()},
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--exp", type=int, default=4)
    args = ap.parse_args()

    device = _get_device()
    logger.info("Device: %s | Exp %d", device, args.exp)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_results = {}
    for m in MODELS:
        if args.model and m != args.model:
            continue
        exp_dir = os.path.join(base, "experiments", f"exp{args.exp}_sphor_ncd_{m}")
        if not os.path.isdir(exp_dir):
            logger.warning("Skip %s: dir mancante", exp_dir)
            continue
        all_results[m] = analyze_model(exp_dir, device, args.exp)

    if len(all_results) > 1:
        line = "\n=== RIEPILOGO GCD (Exp %d · SS-KM vincolato; ACC sui soli test) ===" % args.exp
        logger.info(line)
        logger.info("%-16s %25s %25s", "", "SS-KMeans (Vaze'22) test", "KMeans baseline test")
        logger.info("%-16s %8s %7s %7s %8s %7s %7s", "model", "All", "Old", "New", "All", "Old", "New")
        for m, r in all_results.items():
            ss, km = r["ss_kmeans"], r["kmeans"]
            logger.info(
                "%-16s %8.3f %7.3f %7.3f %8.3f %7.3f %7.3f",
                m,
                ss["acc_all_test"],
                ss["acc_old_test"],
                ss["acc_new_test"],
                km["acc_all_mean"],
                km["acc_old_mean"],
                km["acc_new_mean"],
            )
        summary_path = os.path.join(base, "experiments", f"gcd_summary_exp{args.exp}.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("\nSalvato: %s", summary_path)


if __name__ == "__main__":
    main()
