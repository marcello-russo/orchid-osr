import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter(prefix="/api/ncd", tags=["ncd"])

_ALLOWED_MODELS = {"dinov2_small", "dinov2_base", "resnet18", "resnet50",
                   "convnext_tiny", "convnext_small"}


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


@router.get("/{model_variant}/results")
async def get_ncd_results(model_variant: str, exp: int = 8):
    exp_dir = _get_exp_dir(model_variant, exp)
    if not exp_dir:
        raise HTTPException(404, f"Exp {exp} model '{model_variant}' not found")
    ncd_path = os.path.join(exp_dir, "ncd_results.json")
    if not os.path.exists(ncd_path):
        raise HTTPException(404, "NCD results not found — run --ncd pipeline first")
    with open(ncd_path) as f:
        return json.load(f)


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
