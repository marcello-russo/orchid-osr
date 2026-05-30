# app/api/routes/species.py
import json
from pathlib import Path

from fastapi import APIRouter

from app.api.schemas import SpeciesInfo, SpeciesListResponse
from app.config import settings

router = APIRouter()

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Pre-computed species counts (shipped with the app)
_COUNTS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "species_counts.json"


def _get_dataset_dir() -> Path:
    """Resolve dataset directory at runtime (not import time)."""
    base = Path(settings.dataset_dir)
    for subdir in ["combined", "train_clean", "train"]:
        candidate = base / subdir
        if candidate.is_dir():
            return candidate
    return base


@router.get("/api/species", response_model=SpeciesListResponse)
async def list_species():
    """List all species in the dataset with image counts."""
    # Try filesystem first (local dev with full dataset)
    dataset_dir = _get_dataset_dir()
    if dataset_dir.is_dir():
        species_list: list[SpeciesInfo] = []
        total_images = 0
        for species_dir in sorted(dataset_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            count = sum(1 for f in species_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
            if count > 0:
                species_list.append(SpeciesInfo(name=species_dir.name, image_count=count))
                total_images += count
        return SpeciesListResponse(
            species=species_list,
            total_images=total_images,
            total_species=len(species_list),
        )

    # Fallback: pre-computed counts (deployed environments)
    if _COUNTS_FILE.exists():
        with open(_COUNTS_FILE) as f:
            data = json.load(f)
        return SpeciesListResponse(
            species=[SpeciesInfo(name=k, image_count=v) for k, v in data["species"].items()],
            total_images=data["total_images"],
            total_species=data["total_species"],
        )

    return SpeciesListResponse(species=[], total_images=0, total_species=0)
