"""Exp 6 (pilota): protocollo OSR+GCD della tesi su intenti testuali (CLINC150).

Specchio dello Studio II-A con dominio diverso: le query out-of-scope (oos,
assenti dal train) sono la novel nascosta, come *O. garganica*.
Stesse fasi, stessi moduli del backend OrchID:
  Fase 1 SupCon (SupConLoss tau=0.07, proiezione 384->128, backbone MiniLM congelato)
  Fase 2 linear probe CE su proiezioni congelate (+ variante SpHOR)
  OSR: Mahalanobis, KNN k=5, Energy, Ensemble (media ranghi) + DeLong
  GCD: SS-KM vincolato vs K-means x10, ACC old/new ungherese, NMI/ARI
Pilot a singolo seed (57), CPU. Output: backend/experiments/exp6_clinc_osr/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments" / "exp6_clinc_osr"

from app.core.losses import SphericalPrototypes, SupConLoss  # noqa: E402
from app.core.ncd.clustering import ss_kmeans_multirun  # noqa: E402
from app.core.ncd.delong import delong_auc_ci, delong_paired_test  # noqa: E402
from app.core.ncd.detector import EnergyDetector, EnsembleDetector, KNNDetector, MahalanobisDetector  # noqa: E402
from app.core.ncd.evaluation import hungarian_accuracy  # noqa: E402

SEED = 57
PROJ_DIM = 128
EPOCHS_P1 = 30
EPOCHS_P2 = 15
BATCH = 512


def encode(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(
        model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True),
        dtype=np.float32,
    )


def train_head(X: np.ndarray, y: np.ndarray, sphor: bool = False):
    torch.manual_seed(SEED)
    n_cls = int(y.max()) + 1
    head = nn.Sequential(nn.Linear(X.shape[1], PROJ_DIM))
    crit = SupConLoss(temperature=0.07)
    protos = SphericalPrototypes(n_cls, PROJ_DIM) if sphor else None
    params = list(head.parameters()) + (list(protos.parameters()) if protos else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    Xt = torch.from_numpy(X)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, torch.from_numpy(y)),
        batch_size=BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    for _ in range(EPOCHS_P1):
        for xb, yb in loader:
            zb = torch.nn.functional.normalize(head(xb), dim=1)
            loss = crit(zb, yb)
            if protos is not None:
                # Accoppiata vera (manca nel backend: la repulsione da sola
                # non tocca le rappresentazioni): CE sui prototipi + repulsione.
                sim = protos(zb)
                ce_proto = torch.nn.functional.cross_entropy(sim * 10.0, yb)
                loss = loss + ce_proto + 0.01 * protos.repulse_loss()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        Z = torch.nn.functional.normalize(head(Xt), dim=1).numpy()
    return head, Z


def train_probe(Z: np.ndarray, y: np.ndarray):
    torch.manual_seed(SEED)
    probe = nn.Linear(PROJ_DIM, int(y.max()) + 1)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Z), torch.from_numpy(y)),
        batch_size=BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    for _ in range(EPOCHS_P2):
        for zb, yb in loader:
            loss = crit(probe(zb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe


def project(head, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.nn.functional.normalize(head(torch.from_numpy(X)), dim=1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sphor", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("clinc_oos", "plus")
    names = ds["test"].features["intent"].names
    oos_id = names.index("oos")
    tr = ds["train"]
    te = ds["test"]
    # Protocollo chiuso: oos esclusa dall'addestramento (come garganica).
    # Il train CLINC contiene 250 righe oos: si scartano.
    Xtr_txt = [r["text"] for r in tr if r["intent"] != oos_id]
    ytr_raw = np.array([r["intent"] for r in tr if r["intent"] != oos_id])
    # Rimappa a 0..149 contigui (oos=42 lascia un buco negli id originali)
    known_ids = sorted(set(ytr_raw.tolist()))
    remap = {old: new for new, old in enumerate(known_ids)}
    ytr = np.array([remap[v] for v in ytr_raw])
    yte = np.array([remap.get(v, 150) for v in te["intent"]])
    N_NOVEL_CLS = 150
    Xte_txt = te["text"]
    if args.limit_train:
        idx = rng.choice(len(Xtr_txt), args.limit_train, replace=False)
        Xtr_txt = [Xtr_txt[i] for i in idx]
        ytr = ytr[idx]
    print(
        f"train {len(Xtr_txt)} (150 intenti, oos assente) | "
        f"test {len(Xte_txt)} (oos={int((yte == N_NOVEL_CLS).sum())})",
        flush=True,
    )

    print("encoding MiniLM…", flush=True)
    Etr = encode(Xtr_txt)
    Ete = encode(Xte_txt)

    print("fase 1 SupCon…", flush=True)
    head, Ztr = train_head(Etr, ytr, sphor=args.sphor)
    print("fase 2 probe…", flush=True)
    probe = train_probe(Ztr, ytr)
    Zte = project(head, Ete)
    with torch.no_grad():
        logits_te = probe(torch.from_numpy(Zte)).numpy()

    y_true = (yte == N_NOVEL_CLS).astype(int)
    dets = {
        "Mahalanobis": MahalanobisDetector(),
        "KNN": KNNDetector(k=5),
        "Energy": EnergyDetector(),
        "Ensemble": EnsembleDetector(),
    }
    res_det, scores = {}, {}
    for name, det in dets.items():
        det.fit(Ztr, ytr)
        s = np.asarray(det.score(Zte, logits=logits_te), dtype=float)
        scores[name] = s
        fpr, tpr, _ = roc_curve(y_true, s)
        ci = delong_auc_ci(y_true, s)
        res_det[name] = {
            "auroc": round(float(roc_auc_score(y_true, s)), 4),
            "auroc_ci95": [round(ci["ci_low"], 4), round(ci["ci_high"], 4)],
            "tpr_at_fpr5": round(float(np.interp(0.05, fpr, tpr)), 4),
        }
        j = int(np.searchsorted(tpr, 0.95))
        res_det[name]["fpr_at_tpr95"] = round(float(fpr[min(j, len(fpr) - 1)]), 4)
    r = delong_paired_test(y_true, scores["KNN"], scores["Energy"])
    res_det["delong_knn_vs_energy"] = {"diff": round(r["diff"], 4), "p": round(r["p_value"], 4)}

    # GCD: pool = train (labeled) + test (unlabeled); novel = oos -> classe 150
    print("GCD SS-KM…", flush=True)
    pool = np.concatenate([Ztr, Zte])
    mask = np.concatenate([np.ones(len(Ztr), bool), np.zeros(len(Zte), bool)])
    lab = np.concatenate([ytr, np.full(len(Zte), -1)])
    ss_labels, _ = ss_kmeans_multirun(pool, mask, lab, n_known=150, n_novel=1, n_init=5, max_iter=50, random_state=SEED)
    ss_test = ss_labels[~mask]
    yte_151 = yte  # gia rimappato: note 0..149, novel 150
    acc_old = hungarian_accuracy(yte_151[yte_151 != 150], ss_test[yte_151 != 150])
    acc_new = hungarian_accuracy(yte_151[yte_151 == 150], ss_test[yte_151 == 150])
    print("GCD K-means x10…", flush=True)
    km = KMeans(n_clusters=151, n_init=10, random_state=SEED).fit_predict(Zte)
    km_new = hungarian_accuracy(yte_151[yte_151 == 150], km[yte_151 == 150])
    km_old = hungarian_accuracy(yte_151[yte_151 != 150], km[yte_151 != 150])
    res_gcd = {
        "ss_km": {
            "acc_old": round(float(acc_old), 4),
            "acc_new": round(float(acc_new), 4),
            "nmi_pool_test": round(float(normalized_mutual_info_score(yte_151, ss_test)), 4),
            "ari_pool_test": round(float(adjusted_rand_score(yte_151, ss_test)), 4),
        },
        "kmeans_x10": {"acc_old": round(float(km_old), 4), "acc_new": round(float(km_new), 4)},
    }

    out = {
        "sphor": args.sphor,
        "seed": SEED,
        "minutes": round((time.time() - t0) / 60, 1),
        "detectors": res_det,
        "gcd": res_gcd,
    }
    (OUT / ("results_sphor.json" if args.sphor else "results.json")).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
