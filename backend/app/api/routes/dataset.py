import mimetypes
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.api.deps import require_role, user_from_header_or_query
from app.api.schemas import ClassImagesResponse, DatasetClassesResponse, DatasetClassInfo, DatasetImage
from app.config import settings

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VALID_SPLITS = {"train", "test"}
_VALID_CLASSES = set(settings.class_names)
_CROPPED_DIR = "cropped_256x512"
_HF_DATASET = "marcellorusso/orchid-ncd-dataset"
_HF_BRANCH = "main"


def _hf_token() -> str | None:
    if settings.hf_token:
        return settings.hf_token
    secret_path = "/run/secrets/HF_TOKEN"
    if Path(secret_path).exists():
        with open(secret_path) as f:
            return f.read().strip()
    return None


def _get_dataset_base() -> Path:
    return Path(settings.dataset_dir) / _CROPPED_DIR


def _validate_segment(value: str) -> str:
    """Reject path traversal in URL segments."""
    if ".." in value or "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail="Invalid path segment")
    return value


@router.get("/classes", response_model=DatasetClassesResponse)
async def list_classes(user=Depends(require_role("user"))):
    """List all classes with train/test counts."""
    base = _get_dataset_base()
    classes: list[DatasetClassInfo] = []
    total_train = 0
    total_test = 0

    train_dir = base / "train"
    test_dir = base / "test"

    if not train_dir.is_dir() and not test_dir.is_dir():
        return DatasetClassesResponse(classes=[], total_train=0, total_test=0)

    # Discover classes from both train and test (a class may exist only in test, e.g. NCD novel class)
    seen = set()
    for base_dir in [train_dir, test_dir]:
        if not base_dir.is_dir():
            continue
        for class_dir in sorted(base_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name in seen:
                continue
            seen.add(class_dir.name)
            class_name = class_dir.name
            train_count = (
                sum(1 for f in (train_dir / class_name).iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
                if (train_dir / class_name).is_dir()
                else 0
            )
            test_count = (
                sum(1 for f in (test_dir / class_name).iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
                if (test_dir / class_name).is_dir()
                else 0
            )
            if train_count + test_count > 0:
                classes.append(
                    DatasetClassInfo(
                        name=class_name,
                        train_count=train_count,
                        test_count=test_count,
                        total=train_count + test_count,
                    )
                )
                total_train += train_count
                total_test += test_count

    return DatasetClassesResponse(classes=classes, total_train=total_train, total_test=total_test)


@router.get("/classes/{class_name}/images", response_model=ClassImagesResponse)
async def list_class_images(
    class_name: str,
    split: str = "train",
    limit: int = 12,
    offset: int = 0,
    user=Depends(require_role("user")),
):
    """List images for a class in a given split."""
    class_name = _validate_segment(class_name)
    if class_name not in _VALID_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown class: {class_name}")
    if split not in _VALID_SPLITS:
        raise HTTPException(status_code=400, detail="Invalid split. Use: train or test")

    base = _get_dataset_base()
    class_dir = base / split / class_name

    if not class_dir.is_dir():
        return ClassImagesResponse(class_name=class_name, split=split, total=0, images=[])

    all_files = sorted(f.name for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
    total = len(all_files)
    page = all_files[offset : offset + limit]

    images = [DatasetImage(filename=f, url=f"/api/dataset/images/{split}/{class_name}/{f}") for f in page]
    return ClassImagesResponse(class_name=class_name, split=split, total=total, images=images)


async def _hf_proxy(class_name: str, filename: str) -> Response:
    """Proxy image from the HuggingFace dataset CDN via the backend."""
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # Try all crop directories the image could live in
    for crop in [
        "cropped/224x224/train",
        "cropped/224x224/test",
        "cropped/256x512/train",
        "cropped/256x512/test",
    ]:
        hf_url = f"https://huggingface.co/datasets/{_HF_DATASET}/resolve/{_HF_BRANCH}/{crop}/{class_name}/{filename}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(hf_url, headers=headers)
            if resp.status_code == 200:
                return Response(content=resp.content, media_type=resp.headers.get("content-type", "image/jpeg"))
    raise HTTPException(status_code=404, detail="Image not found locally or on HuggingFace")


async def _serve_resized(file_path: Path, size: int, base: Path, split: str, class_name: str, filename: str):
    """Generate and serve a resized WebP thumbnail."""
    cache_dir = base / ".thumbnails" / str(size) / split / class_name
    cache_path = cache_dir / f"{file_path.stem}.webp"
    if cache_path.is_file():
        return FileResponse(cache_path, media_type="image/webp")

    def _generate():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(file_path) as img:
                w, h = img.size
                if w > size:
                    new_h = int(h * (size / w))
                    img = img.resize((size, new_h), Image.Resampling.BILINEAR)
                img.save(cache_path, "WEBP", quality=80)
            return True
        except Exception as e:
            print(f"Thumbnail generation failed: {e}")
            return False

    success = await run_in_threadpool(_generate)
    if success and cache_path.is_file():
        return FileResponse(cache_path, media_type="image/webp")
    media_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


@router.get("/images/{split}/{class_name}/{filename}")
async def serve_image(
    split: str,
    class_name: str,
    filename: str,
    size: int = Query(None, gt=0, le=1024),
    user=Depends(user_from_header_or_query),
):
    """Serve a dataset image file, optionally resized and converted to WebP."""
    split = _validate_segment(split)
    class_name = _validate_segment(class_name)
    filename = _validate_segment(filename)
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
        return await _hf_proxy(class_name, filename)

    if size is None:
        media_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
        return FileResponse(file_path, media_type=media_type)

    return await _serve_resized(file_path, size, base, split, class_name, filename)
