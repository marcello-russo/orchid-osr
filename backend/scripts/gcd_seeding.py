"""GCD novel-centroid seeding — strategie A/B/C/D + baseline-rerun (spec 2026-09-14 §9).

Protocollo pre-registrato (spec §3): K=6, pool 1929 (1629 train labeled +
300 test unlabeled), clustering sul pool, mapping Hungarian sul pool,
ACC old/new/all sulle 300 test. Baseline-rerun (20 init best-inertia,
seed 57+i) appaiato con A (stessi candidati, best-silhouette, 10 seed
globali 57..66); B descrittivo (density, 1 run); C descrittivo
(KMeans test-only, best-vs-best).

Input da experiments/exp4_*/embeddings/: embeddings.npy (300 test raw),
train_embeddings_known.npy (1629), train_embeddings_known_species.json,
metadata.json points[].species (fonte specie test — VIETATO labels.npy).

Solo CPU/numpy/sklearn, niente training.
Uso:
  python scripts/gcd_seeding.py --exp 4                  # tutte le 6 architetture
  python scripts/gcd_seeding.py --exp 4 resnet18         # una sola
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cli.definitions import resolve_experiment
from app.core.ncd.clustering import (
    _constrained_lloyd,
    build_labeled_mask,
    density_novel_init,
    osr_guided_novel_init,
    ss_kmeans_multirun,
)
from app.core.ncd.evaluation import (
    compute_ari,
    compute_nmi,
    gcd_old_new_all_acc,
    pool_hungarian_test_acc,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]

ALL_SPECIES = sorted(
    ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]
)

GLOBAL_SEEDS = list(range(57, 67))
N_INIT = 20


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _eval_ss(pool_true: np.ndarray, pred: np.ndarray, n_train: int, test_known_mask: np.ndarray) -> dict:
    """ACC test (mapping pool) + NMI/ARI descrittive sul pool."""
    out = pool_hungarian_test_acc(pool_true, pred, n_train, test_known_mask)
    out["mapping"] = "pool"  # mapping Hungarian fittato sul pool, slice test (spec §3)
    out["nmi_pool"] = compute_nmi(pool_true, pred)
    out["ari_pool"] = compute_ari(pool_true, pred)
    return out


def _per_novel(
    pool_true: np.ndarray,
    pred: np.ndarray,
    n_train: int,
    test_species_arr: np.ndarray,
    novel_names: list[str],
) -> dict:
    """Frazione corretta per specie novel (stesso mapping pool, slice test)."""
    from app.core.ncd.evaluation import hungarian_mapping

    mapping = hungarian_mapping(pool_true, pred)
    idx_test = np.arange(n_train, n_train + len(test_species_arr))
    mapped_test = np.array([mapping.get(int(p), -1) for p in pred[idx_test]])
    correct_test = mapped_test == pool_true[idx_test]
    return {
        sp: float(correct_test[test_species_arr == sp].mean()) if (test_species_arr == sp).sum() else float("nan")
        for sp in novel_names
    }


def load_osr_novelty_scores(exp_dir: str, key: str = "KNN") -> np.ndarray:
    """Carica SOLO l'array score OSR ``key`` da ``osr_scores.npz`` (spec §9).

    Barriera anti-leakage strutturale: la funzione accetta solo il nome
    dello score e non espone alcun parametro label; ``y_true`` non viene
    mai letto nel path di selezione (un solo accesso ``z[key]``).
    """
    path = os.path.join(exp_dir, "osr_scores.npz")
    with np.load(path) as z:
        scores = np.asarray(z[key], dtype=np.float64).ravel()
    return scores


def analyze_model(exp_dir: str, exp_id: int) -> dict | None:
    model_variant = os.path.basename(exp_dir).replace(f"exp{exp_id}_sphor_ncd_", "")
    novel_names = list(resolve_experiment(exp_id)["exclude_classes"])
    n_novel = len(novel_names)
    known_classes = [c for c in ALL_SPECIES if c not in novel_names]
    n_known = len(known_classes)
    known_to_idx = {c: i for i, c in enumerate(known_classes)}

    emb_dir = os.path.join(exp_dir, "embeddings")
    # ---- Test (300, raw): specie SOLO da metadata.json ----
    test_emb_raw = np.load(os.path.join(emb_dir, "embeddings.npy"))
    metadata = json.load(open(os.path.join(emb_dir, "metadata.json")))
    test_species = [p["species"] for p in metadata["points"]]
    if not (len(test_species) == len(test_emb_raw) == 300):
        raise ValueError(f"test inatteso: specie={len(test_species)} emb={test_emb_raw.shape}")
    test_emb = l2norm(test_emb_raw)
    if not np.allclose(np.linalg.norm(test_emb, axis=1), 1.0, atol=1e-5):
        raise ValueError("normalizzazione L2 test fallita")
    test_true = np.array(
        [known_to_idx[s] if s in known_to_idx else n_known + novel_names.index(s) for s in test_species]
    )
    test_species_arr = np.array(test_species)
    test_known_mask = np.array([s in known_to_idx for s in test_species])

    # ---- Train known labeled (1629): label da tabella nomi esplicita ----
    train_emb_raw = np.load(os.path.join(emb_dir, "train_embeddings_known.npy"))
    train_species = json.load(open(os.path.join(emb_dir, "train_embeddings_known_species.json")))
    if len(train_species) != len(train_emb_raw):
        raise ValueError(f"train inatteso: specie={len(train_species)} emb={train_emb_raw.shape}")
    train_emb = l2norm(train_emb_raw)
    if not np.allclose(np.linalg.norm(train_emb, axis=1), 1.0, atol=1e-5):
        raise ValueError("normalizzazione L2 train fallita")
    train_labels = np.array([known_to_idx[s] for s in train_species])

    if exp_id == 4 and not (len(test_species) == 300 and test_known_mask.sum() == 250):
        raise ValueError("protocollo Exp4 violato: attesi 300 test di cui 250 known")

    # ---- Pool GCD ----
    pool_emb = np.vstack([train_emb, test_emb])
    n_train = len(train_emb)
    labeled_mask = build_labeled_mask(n_train, len(test_emb))
    pool_true = np.concatenate([train_labels, test_true])
    unlabeled_idx = np.where(~labeled_mask)[0]

    out: dict = {
        "protocol": f"GCD seeding (spec 2026-09-14) — Exp {exp_id}, {n_known} note + {n_novel} novel",
        "novel_classes": novel_names,
        "n_labeled": int(n_train),
        "n_unlabeled": int(len(test_emb)),
        "n_pool": int(len(pool_emb)),
        "n_init": N_INIT,
        "global_seeds": GLOBAL_SEEDS,
        "l2_normalized": True,
    }

    # ---- Baseline-rerun: 20 init best-inertia, disegno appaiato su 10 seed globali ----
    # Per ogni g gli init usano random_state g+i: stessi candidati e seed di A
    # (A riseleziona best-silhouette sugli STESSI candidati). g=57 condivide
    # i candidati con A per costruzione; gli altri g rigirano il proprio rerun.
    base_seeds = []
    base_cands = None
    for g in GLOBAL_SEEDS:
        pred_g, info_g = ss_kmeans_multirun(
            pool_emb,
            labeled_mask,
            train_labels,
            n_known=n_known,
            n_novel=n_novel,
            random_state=g,
            n_init=N_INIT,
            criterion="inertia",
            return_candidates=True,
        )
        cands_g = info_g.pop("candidates")
        if g == 57:
            base_cands = cands_g
            pred_base, info_base = pred_g, info_g
        base_seeds.append(
            {
                **_eval_ss(pool_true, pred_g, n_train, test_known_mask),
                "per_novel_test": _per_novel(pool_true, pred_g, n_train, test_species_arr, novel_names),
                "seed": g,
                "inertia": info_g["inertia"],
                "best_init": info_g["best_init"],
                "n_iter": info_g["n_iter"],
                "inertias": [c["inertia"] for c in cands_g],
            }
        )
    acc_new_base = np.array([s["acc_new_test"] for s in base_seeds])
    out["baseline_rerun"] = {
        **_eval_ss(pool_true, pred_base, n_train, test_known_mask),
        "per_novel_test": _per_novel(pool_true, pred_base, n_train, test_species_arr, novel_names),
        "seed": 57,
        "inertia": info_base["inertia"],
        "best_init": info_base["best_init"],
        "n_iter": info_base["n_iter"],
        "inertias": [c["inertia"] for c in base_cands],
    }
    out["baseline_reruns"] = {
        "seeds": base_seeds,
        "acc_new_mean": float(acc_new_base.mean()),
        "acc_new_std": float(acc_new_base.std()),
    }

    # ---- A: stessi candidati, best-silhouette; 10 seed globali ----
    a_seeds = []
    for g in GLOBAL_SEEDS:
        if g == 57:
            cands = base_cands  # candidati condivisi col baseline-rerun (disegno appaiato)
            pred_a, info_a = pred_base, {**info_base, "candidates": base_cands}
            # riselezione silhouette sugli stessi candidati
            from app.core.ncd.clustering import _best_by_silhouette

            best_idx, sils, discarded, fallback = _best_by_silhouette(pool_emb[unlabeled_idx], cands, n_known + n_novel)
            pred_a = cands[best_idx]["labels"]
            info_a = {
                "silhouette": float(sils[best_idx]),
                "silhouettes": [float(s) for s in sils],
                "discarded": [bool(x) for x in discarded],
                "n_discarded": int(sum(1 for x in discarded if x)),
                "fallback_alldiscarded": bool(fallback),
                "best_init": int(best_idx),
            }
        else:
            pred_a, info_a = ss_kmeans_multirun(
                pool_emb,
                labeled_mask,
                train_labels,
                n_known=n_known,
                n_novel=n_novel,
                random_state=g,
                n_init=N_INIT,
                criterion="silhouette",
                return_candidates=True,
            )
            cands = info_a.pop("candidates")
        # distribuzione ACC new sui candidati non scartati
        acc_new_cands = []
        for i, c in enumerate(cands):
            if info_a["discarded"][i]:
                continue
            acc_new_cands.append(
                pool_hungarian_test_acc(pool_true, c["labels"], n_train, test_known_mask)["acc_new_test"]
            )
        seed_rec = {
            **_eval_ss(pool_true, pred_a, n_train, test_known_mask),
            "per_novel_test": _per_novel(pool_true, pred_a, n_train, test_species_arr, novel_names),
            "seed": g,
            "inertia": float(cands[info_a["best_init"]]["inertia"]),
            "best_init": info_a["best_init"],
            "silhouette": info_a["silhouette"],
            "silhouettes": info_a["silhouettes"],
            "discarded": info_a["discarded"],
            "n_discarded": info_a["n_discarded"],
            "fallback_alldiscarded": info_a["fallback_alldiscarded"],
            "acc_new_candidates": acc_new_cands,
            "acc_new_min": float(np.min(acc_new_cands)) if acc_new_cands else float("nan"),
            "acc_new_med": float(np.median(acc_new_cands)) if acc_new_cands else float("nan"),
            "acc_new_max": float(np.max(acc_new_cands)) if acc_new_cands else float("nan"),
            "inertias": [c["inertia"] for c in cands],
        }
        a_seeds.append(seed_rec)
    acc_new_sel = np.array([s["acc_new_test"] for s in a_seeds])
    out["A_silhouette"] = {
        "seeds": a_seeds,
        "acc_new_mean": float(acc_new_sel.mean()),
        "acc_new_std": float(acc_new_sel.std()),
    }

    # ---- B: density seeding, 1 run deterministico (solo 1 novel: Exp4) ----
    if n_novel != 1:
        out["B_density"] = {
            "skipped": f"n_novel={n_novel} != 1: density seeding definito solo per 1 novel (Exp4)",
            "mapping": "pool",
        }
    else:
        labeled_pool = pool_emb[labeled_mask]
        known_centroids = np.zeros((n_known, pool_emb.shape[1]))
        for c in range(n_known):
            known_centroids[c] = labeled_pool[train_labels == c].mean(axis=0)
        novel_centroid, dens_info = density_novel_init(
            pool_emb,
            unlabeled_idx,
            known_centroids,
            k=10,
            pct=90,
            rng=np.random.default_rng(57),
        )
        init_b = np.vstack([known_centroids, novel_centroid[None, :]])
        pred_b, _, inertia_b, n_iter_b = _constrained_lloyd(pool_emb, labeled_mask, train_labels, init_b)
        out["B_density"] = {
            **_eval_ss(pool_true, pred_b, n_train, test_known_mask),
            "per_novel_test": _per_novel(pool_true, pred_b, n_train, test_species_arr, novel_names),
            "fallback_nosoglia": dens_info["fallback_nosoglia"],
            "threshold": dens_info["threshold"],
            "seed_unlabeled_pos": dens_info["seed_unlabeled_pos"],
            "k": dens_info["k"],
            "pct": dens_info["pct"],
            "inertia": inertia_b,
            "n_iter": n_iter_b,
        }

    # ---- D: init guidato OSR (spec §9), 1 run deterministico (solo 1 novel: Exp4) ----
    if n_novel != 1:
        out["D_osr_guided"] = {
            "skipped": f"n_novel={n_novel} != 1: OSR-guided seeding definito solo per 1 novel (Exp4)",
            "mapping": "pool",
        }
    else:
        labeled_pool = pool_emb[labeled_mask]
        known_centroids_d = np.zeros((n_known, pool_emb.shape[1]))
        for c in range(n_known):
            known_centroids_d[c] = labeled_pool[train_labels == c].mean(axis=0)
        novelty_scores = load_osr_novelty_scores(exp_dir, key="KNN")
        if novelty_scores.shape[0] != len(unlabeled_idx):
            raise ValueError(f"KNN OSR scores inattesi: {novelty_scores.shape[0]} != {len(unlabeled_idx)} unlabeled")
        novel_centroid_d, osr_info = osr_guided_novel_init(
            pool_emb,
            unlabeled_idx,
            known_centroids_d,
            novelty_scores,
            top_k=50,
        )
        init_d = np.vstack([known_centroids_d, novel_centroid_d[None, :]])
        pred_d, _, inertia_d, n_iter_d = _constrained_lloyd(pool_emb, labeled_mask, train_labels, init_d)
        out["D_osr_guided"] = {
            **_eval_ss(pool_true, pred_d, n_train, test_known_mask),
            "per_novel_test": _per_novel(pool_true, pred_d, n_train, test_species_arr, novel_names),
            "detector": "KNN",
            "top_k": osr_info["top_k"],
            "top50_idx": osr_info["top50_idx"],
            "top50_pos": osr_info["top50_pos"],
            "inertia": inertia_d,
            "n_iter": n_iter_d,
        }

    # ---- C: KMeans test-only, 10 init SeedSequence, best-vs-best ----
    # Vince impl §2: il clustering C vive SOLO sul test, quindi il mapping
    # Hungarian è necessariamente fittato sul test (gcd_old_new_all_acc su
    # test_true). Il divieto di gcd_old_new_all_acc-sul-test vale per A/B
    # (pool 1929), dove si usa pool_hungarian_test_acc. Chiavi acc_* duplicate
    # come acc_*_test per coerenza con le altre sezioni; mapping="test".
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    c_results = []
    for g in GLOBAL_SEEDS:
        cands = []
        for j in range(10):
            s = int(np.random.SeedSequence(g).spawn(10)[j].generate_state(1)[0]) % 2**31
            km = KMeans(n_clusters=n_known + n_novel, n_init=1, random_state=s)
            lab = km.fit_predict(test_emb)
            try:
                sil = float(silhouette_score(test_emb, lab)) if len(set(lab.tolist())) > 1 else float("nan")
            except ValueError:
                sil = float("nan")
            cands.append({"labels": lab, "inertia": float(km.inertia_), "silhouette": sil, "seed": s})
        inertias = np.array([c["inertia"] for c in cands])
        best_inertia_idx = int(np.argmin(inertias))
        sils = np.array([c["silhouette"] for c in cands])
        best_sil_idx = int(np.nanargmax(sils)) if np.isfinite(sils).any() else best_inertia_idx
        for tag, idx in (("best_inertia", best_inertia_idx), ("best_silhouette", best_sil_idx)):
            m = gcd_old_new_all_acc(test_true, cands[idx]["labels"], test_known_mask)
            c_results.append(
                {
                    "global_seed": g,
                    "criterion": tag,
                    "mapping": "test",
                    "init_idx": idx,
                    "init_seed": cands[idx]["seed"],
                    "inertia": cands[idx]["inertia"],
                    "silhouette": cands[idx]["silhouette"],
                    **{f"{k}": float(v) for k, v in m.items() if k.startswith("acc_")},
                    "acc_all_test": float(m["acc_all"]),
                    "acc_old_test": float(m["acc_old"]),
                    "acc_new_test": float(m["acc_new"]),
                    "nmi": compute_nmi(test_true, cands[idx]["labels"]),
                    "ari": compute_ari(test_true, cands[idx]["labels"]),
                }
            )
    out["C_kmeans_testonly"] = {"runs": c_results}
    for tag in ("best_inertia", "best_silhouette"):
        vals = np.array([r["acc_new_test"] for r in c_results if r["criterion"] == tag])
        out["C_kmeans_testonly"][f"acc_new_mean_{tag}"] = float(vals.mean())
        out["C_kmeans_testonly"][f"acc_new_std_{tag}"] = float(vals.std())

    out_path = os.path.join(exp_dir, "gcd_seeding_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(
        "[%s] base-rerun New=%.3f (10-seed %.3f±%.3f) | A New=%.3f±%.3f | "
        "B New=%s%s | D New=%s | C New sil=%.3f±%.3f inert=%.3f±%.3f",
        model_variant,
        out["baseline_rerun"]["acc_new_test"],
        out["baseline_reruns"]["acc_new_mean"],
        out["baseline_reruns"]["acc_new_std"],
        out["A_silhouette"]["acc_new_mean"],
        out["A_silhouette"]["acc_new_std"],
        ("%.3f" % out["B_density"]["acc_new_test"]) if "acc_new_test" in out["B_density"] else "SKIP",
        " (fallback_nosoglia)" if out["B_density"].get("fallback_nosoglia") else "",
        ("%.3f" % out["D_osr_guided"]["acc_new_test"]) if "acc_new_test" in out["D_osr_guided"] else "SKIP",
        out["C_kmeans_testonly"]["acc_new_mean_best_silhouette"],
        out["C_kmeans_testonly"]["acc_new_std_best_silhouette"],
        out["C_kmeans_testonly"]["acc_new_mean_best_inertia"],
        out["C_kmeans_testonly"]["acc_new_std_best_inertia"],
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--exp", type=int, default=4)
    args = ap.parse_args()

    logger.info("GCD seeding — Exp %d (CPU, no training)", args.exp)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_results = {}
    for m in MODELS:
        if args.model and m != args.model:
            continue
        exp_dir = os.path.join(base, "experiments", f"exp{args.exp}_sphor_ncd_{m}")
        if not os.path.isdir(exp_dir):
            logger.warning("Skip %s: dir mancante", exp_dir)
            continue
        all_results[m] = analyze_model(exp_dir, args.exp)

    if len(all_results) > 1:
        logger.info("\n=== RIEPILOGO SEEDING (Exp %d · ACC new su test) ===", args.exp)
        logger.info("%-16s %8s %16s %8s %8s %16s", "model", "base", "A mean±std", "B", "D", "C sil/inert")
        for m, r in all_results.items():
            b = ("%.3f" % r["B_density"]["acc_new_test"]) if "acc_new_test" in r["B_density"] else "SKIP"
            d = ("%.3f" % r["D_osr_guided"]["acc_new_test"]) if "acc_new_test" in r.get("D_osr_guided", {}) else "SKIP"
            logger.info(
                "%-16s %8.3f %7.3f±%.3f %8s %8s %7.3f/%7.3f",
                m,
                r["baseline_rerun"]["acc_new_test"],
                r["A_silhouette"]["acc_new_mean"],
                r["A_silhouette"]["acc_new_std"],
                b,
                d,
                r["C_kmeans_testonly"]["acc_new_mean_best_silhouette"],
                r["C_kmeans_testonly"]["acc_new_mean_best_inertia"],
            )


if __name__ == "__main__":
    main()
