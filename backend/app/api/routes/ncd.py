import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ncd", tags=["ncd"])

_ALLOWED_MODELS = {"dinov2_small", "dinov2_base", "resnet18", "resnet50", "convnext_tiny", "convnext_small"}

_NP_AVAILABLE = True
try:
    import numpy as np
except ImportError:
    _NP_AVAILABLE = False


def _get_exp_dir(model_variant: str, exp_num: int = 8) -> str | None:
    if model_variant not in _ALLOWED_MODELS:
        return None
    base = Path(settings.results_dir)
    if not base.exists():
        return None
    prefix = f"exp{exp_num}"
    for d in base.iterdir():
        if d.is_dir() and d.name.endswith(model_variant) and d.name.startswith(prefix):
            return str(d)
    return None


def _load_npy_or_none(path: str) -> list | None:
    if not _NP_AVAILABLE:
        return None
    try:
        arr = np.load(path)
        return arr.tolist()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


@router.get("/{model_variant}/results")
async def get_ncd_results(model_variant: str, exp: int = 8):
    exp_dir = _get_exp_dir(model_variant, exp)
    if not exp_dir:
        raise HTTPException(404, f"Exp {exp} model '{model_variant}' not found")
    ncd_path = os.path.join(exp_dir, "ncd_results.json")
    if not os.path.exists(ncd_path):
        raise HTTPException(404, "NCD results not found — run --ncd pipeline first")
    with open(ncd_path) as f:
        data = json.load(f)

    # Enrich with t-SNE coordinates (best-effort, skip if .npy missing)
    emb_dir = os.path.join(exp_dir, "embeddings")
    tsne_2d = _load_npy_or_none(os.path.join(emb_dir, "tsne_2d.npy"))
    tsne_3d = _load_npy_or_none(os.path.join(emb_dir, "tsne_3d.npy"))
    if tsne_2d is not None:
        data["tsne_2d"] = tsne_2d
    if tsne_3d is not None:
        data["tsne_3d"] = tsne_3d

    # Novel class detection
    openset = data.get("openset", {})
    per_class = openset.get("per_class_metrics", [])
    napoli_metrics = next((p for p in per_class if "napoli" in p.get("left_out", "").lower()), None)
    if napoli_metrics:
        data["openset"]["novel_class"] = napoli_metrics

    # Ablation
    es = openset.get("ensemble_scores", {})
    if es.get("mahalanobis") is not None:
        data["openset"]["ablation"] = {
            "mahalanobis": es["mahalanobis"],
            "knn": es.get("knn", 0),
            "energy": es.get("energy", 0),
            "ensemble": openset.get("mean_auroc", 0),
        }

    return data


_ALLOWED_FIGURES = {
    "tsne_known_vs_unknown.png",
    "oscr_curve.png",
    "auroc_per_class.png",
    "clustering_comparison.png",
}


@router.get("/{model_variant}/figure/{figure_name}")
async def get_ncd_figure(model_variant: str, figure_name: str, exp: int = 8):
    exp_dir = _get_exp_dir(model_variant, exp)
    if not exp_dir:
        raise HTTPException(404, f"Exp {exp} model '{model_variant}' not found")
    if figure_name not in _ALLOWED_FIGURES:
        raise HTTPException(400, f"Figure must be one of: {sorted(_ALLOWED_FIGURES)}")
    fig_path = os.path.join(exp_dir, "ncd_figures", figure_name)
    if not os.path.exists(fig_path):
        raise HTTPException(404, "Figure not found — run --ncd pipeline first")
    return FileResponse(fig_path)


@router.get("/comparison")
async def get_ncd_comparison():
    base = Path(settings.results_dir)
    comparison_path = os.path.join(str(base), "ncd_comparison.json")
    if not os.path.exists(comparison_path):
        raise HTTPException(404, "NCD comparison not found — run ncd_comparison.py first")
    with open(comparison_path) as f:
        return json.load(f)
