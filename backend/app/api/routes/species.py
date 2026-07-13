# app/api/routes/species.py
import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import SpeciesInfo, SpeciesListResponse
from app.config import settings

router = APIRouter()

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Pre-computed species counts (shipped with the app)
_COUNTS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "species_counts.json"


def _count_species_files(data_dir: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for species_dir in sorted(data_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        count = sum(1 for f in species_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
        if count > 0:
            result[species_dir.name] = count
    return result


def _get_dataset_dirs() -> list[Path]:
    """Resolve dataset directories at runtime (not import time).

    Returns [train_dir, test_dir] from the first complete directory set found.
    Priority: cropped_256x512 > cropped_224x224 > flat train/test.
    Only ONE crop size is returned to avoid double-counting the same images.
    """
    base = Path(settings.dataset_dir)
    for train_candidate, test_candidate in [
        ("cropped_256x512/train", "cropped_256x512/test"),
        ("cropped_224x224/train", "cropped_224x224/test"),
        ("train", "test"),
    ]:
        train_dir = base / train_candidate
        test_dir = base / test_candidate
        if train_dir.is_dir() or test_dir.is_dir():
            dirs = []
            if train_dir.is_dir():
                dirs.append(train_dir)
            if test_dir.is_dir():
                dirs.append(test_dir)
            return dirs
    # Legacy: combined directory (single-dir format)
    combined = base / "combined"
    if combined.is_dir():
        return [combined]
    return []


@router.get("/api/species", response_model=SpeciesListResponse)
async def list_species():
    """List all species in the dataset with image counts."""
    dataset_dirs = _get_dataset_dirs()
    if dataset_dirs:
        valid = set(settings.class_names)
        all_species: dict[str, int] = {}
        for d in dataset_dirs:
            species_counts = await run_in_threadpool(_count_species_files, d)
            for name, count in species_counts.items():
                if name in valid:
                    all_species[name] = all_species.get(name, 0) + count
        total_images = sum(all_species.values())
        return SpeciesListResponse(
            species=[SpeciesInfo(name=k, image_count=v) for k, v in sorted(all_species.items())],
            total_images=total_images,
            total_species=len(all_species),
        )

    # Fallback: pre-computed counts (deployed environments)
    if _COUNTS_FILE.exists():
        with open(_COUNTS_FILE) as f:
            data = json.load(f)
        valid = set(settings.class_names)
        filtered = {k: v for k, v in data["species"].items() if k in valid}
        return SpeciesListResponse(
            species=[SpeciesInfo(name=k, image_count=v) for k, v in sorted(filtered.items())],
            total_images=sum(filtered.values()),
            total_species=len(filtered),
        )

    return SpeciesListResponse(species=[], total_images=0, total_species=0)
