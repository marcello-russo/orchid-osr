# app/api/routes/species.py
from pathlib import Path

from fastapi import APIRouter

from app.api.schemas import SpeciesInfo, SpeciesListResponse
from app.config import settings

router = APIRouter()

# Image extensions to count
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _get_dataset_dir() -> Path:
    """Resolve dataset directory at runtime (not import time)."""
    combined = Path(settings.dataset_dir) / "combined"
    if combined.is_dir():
        return combined
    return Path(settings.dataset_dir)


@router.get("/api/species", response_model=SpeciesListResponse)
async def list_species():
    """List all species in the dataset with image counts."""
    dataset_dir = _get_dataset_dir()

    species_list: list[SpeciesInfo] = []
    total_images = 0

    for species_dir in sorted(dataset_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        count = sum(
            1 for f in species_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        if count > 0:
            species_list.append(SpeciesInfo(name=species_dir.name, image_count=count))
            total_images += count

    return SpeciesListResponse(
        species=species_list,
        total_images=total_images,
        total_species=len(species_list),
    )
