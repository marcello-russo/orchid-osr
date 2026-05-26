import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import require_role
from app.api.schemas import EmbeddingPoint, EmbeddingResponse, ServingInfo
from app.config import settings

_ALLOWED_EXPERIMENTS: set[str] = set()


def _load_allowed_experiments():
    global _ALLOWED_EXPERIMENTS
    if not _ALLOWED_EXPERIMENTS:
        exp_dir = Path(settings.results_dir)
        if exp_dir.exists():
            _ALLOWED_EXPERIMENTS = {d.name for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("exp")}
    return _ALLOWED_EXPERIMENTS

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@router.get("", response_model=EmbeddingResponse)
async def get_embeddings(
    user=Depends(require_role("user")),
    experiment: str = Query("", description="Experiment directory name (e.g. exp6_clean_split_resnet50)"),
    method: str = Query("tsne_3d", description="Dimensionality reduction: tsne_3d, tsne_2d, umap_2d"),
):
    exp_dir = Path(settings.results_dir)

    if experiment:
        allowed = _load_allowed_experiments()
        if experiment not in allowed:
            raise HTTPException(status_code=400, detail=f"Unknown experiment: {experiment}")
        exp_dir = exp_dir / experiment
    else:
        dirs = sorted(exp_dir.glob("exp*/embeddings/metadata.json"))
        if not dirs:
            raise HTTPException(status_code=404,
                detail="No embeddings found. Train a model first or specify an experiment.")
        exp_dir = dirs[-1].parent

    emb_dir = exp_dir / "embeddings" if exp_dir.name != "embeddings" else exp_dir
    if not emb_dir.exists() or not emb_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No embeddings directory at {emb_dir}")

    metadata = _load_json(emb_dir / "metadata.json")
    if not metadata:
        raise HTTPException(status_code=404, detail="metadata.json not found in embeddings directory")

    coord_map = {"tsne_3d": "tsne_3d.npy", "tsne_2d": "tsne_2d.npy", "umap_2d": "umap_2d.npy", "umap_3d": "umap_3d.npy"}
    coord_file = coord_map.get(method)
    if not coord_file:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    coord_path = emb_dir / coord_file
    if not coord_path.exists():
        raise HTTPException(status_code=404, detail=f"{coord_file} not found")

    import numpy as np
    try:
        coords = np.load(coord_path)
    except (FileNotFoundError, ValueError):
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
                filename=p.get("filename", ""),
                image_url=p.get("image_url", ""),
                split=p.get("split", ""),
            )
            for i, p in enumerate(points_list)
        ]

    display = "tsne_3d" if dim >= 3 else "tsne_2d"

    return EmbeddingResponse(
        experiment=experiment if experiment else exp_dir.name,
        method=display,
        dimension=min(dim, 3),
        total_points=len(points),
        species=species_list,
        points=points,
        serving=ServingInfo(
            note="Embeddings are pre-computed during training and uploaded to HuggingFace. No on-the-fly computation.",
            repo="marcellorusso/orchid-ncd-models",
            path=f"{{exp}}/{experiment}/embeddings/" if experiment else "",
        ),
    )
