# backend/app/api/routes/dataset.py
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.schemas import DatasetClassesResponse, DatasetClassInfo, ClassImagesResponse, DatasetImage
from app.config import settings

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VALID_SPLITS = {"train_clean", "test_clean"}
_SPLIT_ALIASES = {"train": "train_clean", "test": "test_clean"}


def _get_dataset_base() -> Path:
    return Path(settings.dataset_dir)


def _validate_segment(value: str) -> str:
    """Reject path traversal in URL segments."""
    if ".." in value or "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail="Invalid path segment")
    return value


@router.get("/classes", response_model=DatasetClassesResponse)
async def list_classes():
    """List all classes with train/test counts."""
    base = _get_dataset_base()
    classes: list[DatasetClassInfo] = []
    total_train = 0
    total_test = 0

    train_dir = base / "train_clean"
    test_dir = base / "test_clean"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise HTTPException(status_code=404, detail="Dataset directories not found")

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        train_count = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        test_class_dir = test_dir / class_name
        test_count = (
            sum(1 for f in test_class_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
            if test_class_dir.is_dir()
            else 0
        )
        if train_count + test_count > 0:
            classes.append(DatasetClassInfo(
                name=class_name, train_count=train_count,
                test_count=test_count, total=train_count + test_count,
            ))
            total_train += train_count
            total_test += test_count

    return DatasetClassesResponse(classes=classes, total_train=total_train, total_test=total_test)


@router.get("/classes/{class_name}/images", response_model=ClassImagesResponse)
async def list_class_images(class_name: str, split: str = "train", limit: int = 12, offset: int = 0):
    """List images for a class in a given split."""
    class_name = _validate_segment(class_name)
    split = _SPLIT_ALIASES.get(split, split)
    if split not in _VALID_SPLITS:
        raise HTTPException(status_code=400, detail=f"Invalid split. Use: train, test, train_clean, or test_clean")

    base = _get_dataset_base()
    class_dir = base / split / class_name
    if not class_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found in {split}")

    all_files = sorted(
        f.name for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
    )
    total = len(all_files)
    page = all_files[offset: offset + limit]

    images = [
        DatasetImage(filename=f, url=f"/api/dataset/images/{split}/{class_name}/{f}")
        for f in page
    ]
    return ClassImagesResponse(class_name=class_name, split=split, total=total, images=images)


@router.get("/images/{split}/{class_name}/{filename}")
async def serve_image(split: str, class_name: str, filename: str):
    """Serve a dataset image file."""
    split = _validate_segment(split)
    class_name = _validate_segment(class_name)
    filename = _validate_segment(filename)
    split = _SPLIT_ALIASES.get(split, split)
    if split not in _VALID_SPLITS:
        raise HTTPException(status_code=400, detail="Invalid split")

    base = _get_dataset_base()
    file_path = base / split / class_name / filename

    # Ensure resolved path stays within the dataset directory
    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal denied")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    if file_path.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Not an image file")

    media_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
    return FileResponse(file_path, media_type=media_type)
