# app/api/serializers/folds.py
"""Normalizzazione dei fold: formato su disco → formato servito dall'API."""

from typing import Any


def normalize_fold(fold: dict[str, Any]) -> dict[str, Any]:
    """Normalize fold data, separating SupCon and CE phases into fold_phases.

    If fold already has fold_phases (already normalized), return as-is.
    Single-phase (CE baseline, exp6):
      - Truncates history to best_epoch + patience
      - fold_phases = None

    Two-phase (SupCon, exp7+):
      - Phase 1 (contrastive): supcon_history metrics
      - Phase 2 (CE): top-level history metrics (truncated)
      - Returns fold_phases array with Phase 1 data
    """
    if "fold_phases" in fold:
        return fold
    if "supcon_phase" not in fold:
        # Single phase (CE baseline, exp6) — truncate history
        flat = dict(fold)
        total = len(flat.get("history", {}).get("train_loss", []))
        best = flat.get("best_epoch", 0)
        cap = min(best, total - 1) if best > 0 and total > 0 else 0
        cut = min(cap + 15, total) if cap > 0 else total
        if cap > 0 and cut < cap + 1:
            cut = min(cap + 1, total)
        if cut < total:
            h = flat.get("history", {})
            flat["history"] = {k: v[:cut] for k, v in h.items() if isinstance(v, list)}
            flat["best_epoch"] = cut - 1
        flat["fold_phases"] = None
        return flat

    # Two-phase (SupCon, exp7+)
    sp = fold["supcon_phase"]
    sh = sp.get("supcon_history", {})
    ce_hist = fold.get("history", {})

    # Phase 1: SupCon
    supcon_train = sh.get("train_loss", []) or []
    supcon_knn = sh.get("val_knn_acc", []) or []

    # Phase 2: CE
    ce_train = ce_hist.get("train_loss", []) or []
    ce_val_loss = ce_hist.get("val_loss", []) or []
    ce_val_f1 = ce_hist.get("val_f1", []) or []
    ce_val_acc = ce_hist.get("val_accuracy", []) or []

    # Truncate Phase 2 to best_epoch + patience
    n_ce = len(ce_train)
    best_raw = fold.get("best_epoch", 0)
    ce_cap = min(best_raw, n_ce - 1) if best_raw > 0 and n_ce > 0 else 0
    ce_cut = min(ce_cap + 15, n_ce) if ce_cap > 0 else n_ce
    if ce_cap > 0 and ce_cut < ce_cap + 1:
        ce_cut = min(ce_cap + 1, n_ce)

    ce_history = {
        "train_loss": ce_train[:ce_cut],
        "val_loss": ce_val_loss[:ce_cut],
        "val_f1": ce_val_f1[:ce_cut],
        "val_accuracy": ce_val_acc[:ce_cut],
    }
    best_epoch = ce_cut - 1 if ce_cut > 0 else 0

    fold_phases = [
        {
            "name": "SupCon",
            "type": "contrastive",
            "epochs": len(supcon_train),
            "metrics": {
                "train_loss": supcon_train,
                "val_knn_acc": supcon_knn,
            },
        }
    ]

    result = {
        "fold": fold.get("fold", 0),
        "best_f1": fold.get("best_f1", 0),
        "best_accuracy": fold.get("best_accuracy", 0),
        "best_epoch": best_epoch,
        "final_train_loss": ce_history["train_loss"][-1] if ce_history["train_loss"] else 0,
        "final_val_loss": ce_history["val_loss"][-1] if ce_history["val_loss"] else 0,
        "history": ce_history,
        "train_size": fold.get("train_size", 0),
        "val_size": fold.get("val_size", 0),
        "fold_phases": fold_phases,
    }
    # Preserve optional fields from original fold
    for key in ("phase_epochs", "phase_names"):
        if key in fold:
            result[key] = fold[key]
    return result
