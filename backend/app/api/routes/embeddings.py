import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.api.deps import require_role
from app.api.routes.auth import limiter
from app.api.schemas import EmbeddingPoint, EmbeddingResponse, ServingInfo
from app.config import settings

# Vetrina pubblica: un solo esperimento, scelto qui. Le coordinate t-SNE sono
# precalcolate (nessuna inferenza), e il payload pubblico NON contiene filename
# né image_url: il dataset è privato e il proxy immagini serve solo l'anteprima
# di 3 foto per classe.
PUBLIC_EXPERIMENT = "exp4_sphor_ncd_dinov2_small"
PUBLIC_METHOD = "tsne_3d"

_ALLOWED_EXPERIMENTS: set[str] = set()
# Il payload pubblico è statico: costruirlo una volta sola evita di rileggere
# 460 KB di metadata.json a ogni richiesta.
_PUBLIC_CACHE: EmbeddingResponse | None = None


def _load_allowed_experiments() -> set[str]:
    global _ALLOWED_EXPERIMENTS
    if not _ALLOWED_EXPERIMENTS:
        exp_dir = Path(settings.results_dir)
        if exp_dir.exists():
            _ALLOWED_EXPERIMENTS = {d.name for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("exp")}
    return _ALLOWED_EXPERIMENTS


router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            data: Any = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _empty(method: str = PUBLIC_METHOD) -> EmbeddingResponse:
    return EmbeddingResponse(
        experiment="",
        method=method,
        dimension=3,
        total_points=0,
        species=[],
        points=[],
    )


def _build_response(exp_dir: Path, method: str, public: bool) -> EmbeddingResponse:
    emb_dir = exp_dir / "embeddings" if exp_dir.name != "embeddings" else exp_dir
    if not emb_dir.exists() or not emb_dir.is_dir():
        if public:
            return _empty(method)
        raise HTTPException(status_code=404, detail=f"No embeddings directory at {emb_dir}")

    metadata = _load_json(emb_dir / "metadata.json")
    if not metadata:
        if public:
            return _empty(method)
        raise HTTPException(status_code=404, detail="metadata.json not found in embeddings directory")

    coord_map = {"tsne_3d": "tsne_3d.npy", "tsne_2d": "tsne_2d.npy", "umap_2d": "umap_2d.npy", "umap_3d": "umap_3d.npy"}
    coord_file = coord_map.get(method)
    if not coord_file:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    coord_path = emb_dir / coord_file
    if not coord_path.exists():
        if public:
            return _empty(method)
        raise HTTPException(status_code=404, detail=f"{coord_file} not found")

    try:
        coords = np.load(coord_path)
    except (FileNotFoundError, ValueError):
        if public:
            return _empty(method)
        raise HTTPException(status_code=500, detail=f"Failed to load {coord_file} (corrupted?)")

    points_list = metadata.get("points", [])
    if len(points_list) != coords.shape[0]:
        raise HTTPException(status_code=500, detail="Coordinate/metadata length mismatch")

    dim = coords.shape[1] if coords.ndim > 1 else 1
    species_list = sorted(set(p.get("species", "") for p in points_list))

    points = [
        EmbeddingPoint(
            idx=p.get("idx", i),
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            z=float(coords[i, 2]) if dim >= 3 else 0.0,
            species=p.get("species", ""),
            filename="" if public else p.get("filename", ""),
            image_url="" if public else p.get("image_url", ""),
            split="" if public else p.get("split", ""),
        )
        for i, p in enumerate(points_list)
    ]

    display = "tsne_3d" if dim >= 3 else "tsne_2d"

    return EmbeddingResponse(
        experiment=exp_dir.name if public else (exp_dir.name if exp_dir.name != "embeddings" else ""),
        method=display,
        dimension=min(dim, 3),
        total_points=len(points),
        species=species_list,
        points=points,
        serving=ServingInfo(
            note=(
                "Coordinate precalcolate: nessuna inferenza, nessuna immagine del dataset."
                if public
                else (
                    "Embeddings are pre-computed during training and uploaded to HuggingFace. "
                    "No on-the-fly computation."
                )
            ),
            repo="" if public else settings.hf_models_repo,
            path="" if public else f"{{exp}}/{exp_dir.name}/embeddings/",
        ),
    )


@router.get("", response_model=EmbeddingResponse)
async def get_embeddings(
    user: dict[str, Any] = Depends(require_role("user")),
    experiment: str = Query("", description="Experiment directory name (e.g. exp1_ce_resnet50)"),
    method: str = Query("tsne_3d", description="Dimensionality reduction: tsne_3d, tsne_2d, umap_2d"),
) -> EmbeddingResponse:
    base = Path(settings.results_dir)

    if experiment:
        if experiment not in _load_allowed_experiments():
            raise HTTPException(status_code=400, detail=f"Unknown experiment: {experiment}")
        exp_dir = base / experiment
    else:
        dirs = sorted(d for d in base.glob("exp*/embeddings/metadata.json"))
        if not dirs:
            return _empty("tsne_3d")
        exp_dir = dirs[-1].parent

    return _build_response(exp_dir, method, public=False)


@router.get("/public", response_model=EmbeddingResponse)
@limiter.limit("60/minute")
async def get_public_embeddings(request: Request, response: Response) -> EmbeddingResponse:  # noqa: ARG001
    """Spazio latente in sola lettura per la vetrina.

    Un esperimento fisso, coordinate precalcolate, e **nessun** filename o URL
    di immagine: chi non ha un account vede la nuvola colorata per specie, non
    il dataset. Cacheabile perché statico.
    """
    global _PUBLIC_CACHE
    if _PUBLIC_CACHE is None:
        _PUBLIC_CACHE = _build_response(Path(settings.results_dir) / PUBLIC_EXPERIMENT, PUBLIC_METHOD, public=True)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return _PUBLIC_CACHE
