# backend/app/api/routes/experiments.py
import datetime
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import ExperimentDetailResponse, ExperimentSummary, FoldDataResponse
from app.config import settings

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _get_experiments_dir() -> Path:
    return Path(settings.results_dir)


def _validate_id(value: str) -> str:
    """Sanitize path parameter to prevent path traversal."""
    if ".." in value or "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail="Invalid identifier")
    return value


def _parse_exp_num(name: str) -> int:
    """Extract experiment number from directory name for sorting.
    'exp10_mgh_ncd_resnet18' → 10, 'exp6_clean_split' → 6."""
    m = re.search(r'exp(\d+)', name)
    return int(m.group(1)) if m else 999


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments():
    """List all experiments with summary metrics."""
    exp_dir = _get_experiments_dir()
    if not exp_dir.exists():
        return []

    experiments = []
    for d in sorted(exp_dir.iterdir(), key=lambda p: _parse_exp_num(p.name)):
        if not d.is_dir():
            continue

        config_path = d / "config.json"
        results_path = d / "results.json"

        # Need at least config.json or results.json
        if not config_path.exists() and not results_path.exists():
            continue

        config = _load_json(config_path) if config_path.exists() else {}
        results = _load_json(results_path) if results_path.exists() else None

        fold_files = sorted(d.glob("fold_*.json"))
        has_folds = len(fold_files) > 0
        config_type = (config or {}).get("type", "")
        exp_type = "kfold" if has_folds else ("kfold" if config_type == "kfold" else "baseline")

        experiments.append(
            {
                "id": d.name,
                "name": config.get("name", d.name),
                "model_variant": config.get("model_variant"),
                "type": exp_type,
                "status": "completed" if (results or exp_type == "baseline") else "partial",
                "num_folds": len(fold_files) if has_folds else None,
                "test_accuracy": results.get("test_metrics", {}).get("accuracy") if results else None,
                "mean_f1": (results.get("mean_f1") or results.get("test_metrics", {}).get("f1_macro"))
                if results
                else None,
                "std_f1": results.get("std_f1") if results else None,
                "timestamp": config.get("timestamp"),
            }
        )

    return experiments


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(experiment_id: str):
    """Get full experiment detail with all folds."""
    experiment_id = _validate_id(experiment_id)
    exp_dir = _get_experiments_dir() / experiment_id
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    config_path = exp_dir / "config.json"
    results_path = exp_dir / "results.json"

    if not config_path.exists() and not results_path.exists():
        raise HTTPException(status_code=404, detail=f"No config.json or results.json in '{experiment_id}'")

    config = _load_json(config_path) if config_path.exists() else {}

    # Merge recipe info from experiment definitions
    import re
    exp_num_match = re.search(r'exp(\d+)', experiment_id)
    if exp_num_match:
        exp_num = int(exp_num_match.group(1))
        _recipes = {
            6: {
                "phases": ["Cross-Entropy (single phase)"],
                "recipe": "Single-phase training with standard Cross-Entropy loss. "
                          "The entire network (backbone + classifier head) is trained jointly. "
                          "Optimizer: Adam (lr=0.001). Epochs: 150 (early stopping patience=15). "
                          "Batch: 8 (effective 32 via gradient accumulation). "
                          "5-fold stratified cross-validation. "
                          "Purpose: establish a supervised baseline for comparison with contrastive methods.",
            },
            7: {
                "phases": ["Phase 1: SupCon (contrastive pretraining)", "Phase 2: CE fine-tuning (linear probe)"],
                "recipe": "Two-phase training designed to produce structured embeddings for downstream NCD.\n\n"
                          "Phase 1 — SupCon (Supervised Contrastive Learning):\n"
                          "  • Loss function: SupConLoss — pulls same-species embeddings together, "
                          "pushes different-species apart on the hypersphere.\n"
                          "  • Temperature τ = 0.07 (standard value from [8]).\n"
                          "  • Projection head: MLP (embedding_dim → 128) discarded after Phase 1.\n"
                          "  • Backbone + projection head trained jointly.\n"
                          "  • Validation metric: KNN accuracy (k=1) on validation embeddings.\n"
                          "  • Optimizer: AdamW per-architecture (ResNet: lr=0.001, ConvNeXt: lr=0.0001, DINOv2: lr=0.0002).\n\n"
                          "Phase 2 — CE Fine-Tuning (Linear Probe):\n"
                          "  • Backbone is frozen (preserves contrastive structure).\n"
                          "  • Only the linear classifier head is trained with Cross-Entropy loss.\n"
                          "  • Higher learning rate (lr=0.01) since only the randomly initialized head trains.\n"
                          "  • Optimizer: Adam. Max 50 epochs with early stopping (patience=15).\n"
                          "  • Result: classifier that benefits from the structured embedding space.",
            },
            8: {
                "phases": ["Phase 1: SupCon + SpHOR", "Phase 2: CE fine-tuning"],
                "recipe": "Same two-phase structure as Exp 7, with an additional regularization term in Phase 1.\n\n"
                          "Phase 1 — SupCon + SpHOR (Spherical Orthogonal Prototypes):\n"
                          "  • Standard SupCon loss (same as Exp 7) + SpHOR repulsive loss.\n"
                          "  • SpHOR loss: L_rep = (1/K²) Σ_{i≠j} cos(θ_ij)² — forces class prototypes "
                          "to be orthogonal, maximizing angular coverage on the unit hypersphere.\n"
                          "  • This prevents prototype collapse and improves open-set separation.\n"
                          "  • Same optimizers, LR, and scheduling as Exp 7.\n\n"
                          "Phase 2 — CE Fine-Tuning:\n"
                          "  • Identical to Exp 7: frozen backbone, linear classifier head.\n\n"
                          "Evaluation: O. sphegodes_Napoli is the held-out novel class (test-only).",
            },
            9: {
                "phases": ["Phase 1: SupCon + SpHOR", "Phase 2: CE fine-tuning"],
                "recipe": "Structurally identical to Exp 8 (same loss, same hyperparameters).\n"
                          "The only difference: this experiment is configured with ncd_mode=True, "
                          "enabling the full NCD evaluation pipeline (OSR + clustering) "
                          "with O. sphegodes_Napoli as the hidden novel class.",
            },
            10: {
                "phases": ["Phase 1: MGH-SupCon (Multi-Granularity Hard-negative)", "Phase 2: CE fine-tuning"],
                "recipe": "Enhanced two-phase training with two key innovations over Exp 8.\n\n"
                          "Phase 1 — MGH-SupCon:\n"
                          "  • Hard Negative Mining: SupCon loss contributions are weighted per-class pair:\n"
                          "    — (O. majellensis, O. sphegodes): 5× weight (most confused pair).\n"
                          "    — (O. sphegodes, O. sphegodes_Palena): 3× weight.\n"
                          "    — All other pairs: 1× (standard).\n"
                          "  • Multi-Granularity Features (MG):\n"
                          "    — ViT models (DINOv2): concatenate CLS token (global) + "
                          "attention-pooled patch tokens (local) + mid-layer CLS (regional).\n"
                          "    — CNN models (ResNet, ConvNeXt): concatenate penultimate "
                          "and final stage features after global average pooling.\n"
                          "  • Combined embedding dimension: 1152 (DINOv2) or 3072 (ResNet50).\n"
                          "  • Projected to 256-dim via MLP before SupCon loss.\n"
                          "  • Same optimizers and scheduling as Exp 8.\n\n"
                          "Phase 2 — CE Fine-Tuning:\n"
                          "  • Frozen backbone, linear classifier head (same as Exp 7-9).\n\n"
                          "Evaluation: O. sphegodes_Napoli novel class (test-only). "
                          "Expected: improved novel detection via harder embedding separation.",
            },
        }
        if exp_num in _recipes:
            config["phases"] = _recipes[exp_num]["phases"]
            config["recipe"] = _recipes[exp_num]["recipe"]

    folds = []
    for fold_file in sorted(exp_dir.glob("fold_*.json")):
        raw = _load_json(fold_file)
        from app.api.routes.comparison import _normalize_fold
        folds.append(_normalize_fold(raw))

    results = None
    results_path = exp_dir / "results.json"
    if results_path.exists():
        results = _load_json(results_path)

    last_modified = None
    rp = exp_dir / "results.json"
    if rp.exists():
        mtime = rp.stat().st_mtime
        last_modified = datetime.datetime.fromtimestamp(mtime).isoformat()

    return {
        "id": experiment_id,
        "config": config,
        "folds": folds,
        "results": results,
        "last_modified": last_modified,
        "summary": {
            "model_variant": config.get("model_variant"),
            "type": "kfold" if folds else "baseline",
            "num_folds": len(folds),
            "mean_accuracy": results.get("mean_accuracy") if results else None,
            "mean_f1": (results.get("mean_f1") or results.get("test_metrics", {}).get("f1_macro")) if results else None,
            "std_f1": results.get("std_f1") if results else None,
            "test_accuracy": results.get("test_metrics", {}).get("accuracy") if results else None,
            "test_f1": results.get("test_metrics", {}).get("f1_macro") if results else None,
        },
    }


@router.get("/{experiment_id}/log")
async def get_experiment_log(experiment_id: str):
    """Get the training log file for an experiment.
    On HF Space (no log files synced), returns available=false."""
    experiment_id = _validate_id(experiment_id)
    exp_dir = _get_experiments_dir() / experiment_id
    log_file = exp_dir / f"{experiment_id}.log"
    if not log_file.exists():
        return {"log": None, "size": 0, "available": False}
    with open(log_file) as f:
        content = f.read()
    return {"log": content, "size": len(content), "available": True}


@router.get("/{experiment_id}/folds/{fold_number}", response_model=FoldDataResponse)
async def get_fold(experiment_id: str, fold_number: int):
    """Get a single fold with full training history.

    fold_number is 0-based (matching file names: fold_0.json, fold_1.json, ...).
    """
    experiment_id = _validate_id(experiment_id)
    exp_dir = _get_experiments_dir() / experiment_id
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    if fold_number < 0:
        raise HTTPException(status_code=400, detail="fold_number must be >= 0")

    fold_file = exp_dir / f"fold_{fold_number}.json"
    if not fold_file.exists():
        raise HTTPException(status_code=404, detail=f"Fold {fold_number} not found")

    return _load_json(fold_file)
