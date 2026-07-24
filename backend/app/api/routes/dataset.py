import json
import mimetypes
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.api.deps import require_role
from app.api.routes.species import _count_species_files, _get_dataset_dirs
from app.api.schemas import ClassImagesResponse, DatasetClassesResponse, DatasetClassInfo, DatasetImage
from app.config import settings

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VALID_SPLITS = {"train", "test"}
_VALID_CLASSES = set(settings.class_names)
_CROPPED_DIR = "cropped_256x512"
_HF_DATASET = "marcellorusso/orchid-ncd-dataset"
_HF_BRANCH = "main"
_LISTING_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_listing.json"

_http_client: httpx.AsyncClient | None = None
_listing_cache: dict | None = None


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


def _get_listing() -> dict:
    global _listing_cache
    if _listing_cache is not None:
        return _listing_cache
    if _LISTING_PATH.is_file():
        with open(_LISTING_PATH) as f:
            _listing_cache = json.load(f)
        return _listing_cache
    return {"classes": {}, "files": {"train": {}, "test": {}}, "total_images": 0}


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
    if ".." in value or "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail="Invalid path segment")
    return value


@router.get("/classes", response_model=DatasetClassesResponse)
async def list_classes(user=Depends(require_role("user"))):
    listing = _get_listing()
    classes: list[DatasetClassInfo] = []
    total_train = 0
    total_test = 0
    for class_name, counts in sorted(listing.get("classes", {}).items()):
        classes.append(
            DatasetClassInfo(
                name=class_name,
                train_count=counts.get("train", 0),
                test_count=counts.get("test", 0),
                total=counts.get("train", 0) + counts.get("test", 0),
            )
        )
        total_train += counts.get("train", 0)
        total_test += counts.get("test", 0)
    # Fallback: scan filesystem if listing is empty
    if not classes:
        base = _get_dataset_base()
        train_dir = base / "train"
        test_dir = base / "test"
        if train_dir.is_dir() or test_dir.is_dir():
            for base_dir in [train_dir, test_dir]:
                if not base_dir.is_dir():
                    continue
                for class_dir in sorted(base_dir.iterdir()):
                    if not class_dir.is_dir():
                        continue
                    cn = class_dir.name
                    if any(c.name == cn for c in classes):
                        continue
                    train_count = len(list((train_dir / cn).iterdir())) if (train_dir / cn).is_dir() else 0
                    test_count = len(list((test_dir / cn).iterdir())) if (test_dir / cn).is_dir() else 0
                    if train_count + test_count > 0:
                        classes.append(
                            DatasetClassInfo(
                                name=cn,
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
    class_name = _validate_segment(class_name)
    if split not in _VALID_SPLITS:
        raise HTTPException(status_code=400, detail="Invalid split. Use: train or test")

    listing = _get_listing()
    all_files = listing.get("files", {}).get(split, {}).get(class_name, [])

    # Fallback: scan filesystem if listing is empty
    if not all_files:
        base = _get_dataset_base()
        class_dir = base / split / class_name
        if class_dir.is_dir():
            all_files = sorted(f.name for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)

    total = len(all_files)
    page = all_files[offset : offset + limit]
    images = [DatasetImage(filename=f, url=f"/api/dataset/images/{split}/{class_name}/{f}") for f in page]
    return ClassImagesResponse(class_name=class_name, split=split, total=total, images=images)


async def _hf_proxy(class_name: str, filename: str, cache_dir: Path | None = None) -> Response:
    """Proxy image from HuggingFace dataset CDN, optionally caching to disk."""
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for crop in [
        "cropped/224x224/train",
        "cropped/224x224/test",
        "cropped/256x512/train",
        "cropped/256x512/test",
    ]:
        hf_url = f"https://huggingface.co/datasets/{_HF_DATASET}/resolve/{_HF_BRANCH}/{crop}/{class_name}/{filename}"
        client = await _get_http_client()
        resp = await client.get(hf_url, headers=headers, follow_redirects=True)
        if resp.status_code == 200:
            # Cache to disk if requested
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / filename).write_bytes(resp.content)
            content_type = resp.headers.get("content-type", "image/jpeg")
            return Response(content=resp.content, media_type=content_type)
    raise HTTPException(status_code=404, detail="Image not found locally or on HuggingFace")


async def _serve_resized(file_path: Path, size: int, split: str, class_name: str, filename: str):
    """Generate and serve a resized WebP thumbnail, fetching from HF if needed."""
    cache_base = _get_dataset_base().parent / ".thumbnails" / str(size)
    cache_dir = cache_base / split / class_name
    cache_path = cache_dir / f"{file_path.stem}.webp"
    if cache_path.is_file():
        return FileResponse(cache_path, media_type="image/webp")

    # Load image from local file or HF
    if file_path.is_file():
        img_path = file_path
    else:
        img_path = cache_dir / filename
        if not img_path.is_file():
            await _hf_proxy(class_name, filename, cache_dir)
        if not img_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")

    def _generate():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(img_path) as img:
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

    # Fallback: serve original without resize
    if file_path.is_file():
        media_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
        return FileResponse(file_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/{split}/{class_name}/{filename}")
async def serve_image(
    split: str,
    class_name: str,
    filename: str,
    size: int = Query(None, gt=0, le=1024),
):
    """Serve a dataset image file, optionally resized and converted to WebP.

    Checks local filesystem first, then falls back to HuggingFace CDN proxy.
    Images fetched from CDN are cached locally for subsequent requests.
    """
    split = _validate_segment(split)
    class_name = _validate_segment(class_name)
    filename = _validate_segment(filename)
    if split not in _VALID_SPLITS:
        raise HTTPException(status_code=400, detail="Invalid split")

    base = _get_dataset_base()
    file_path = base / split / class_name / filename

    try:
        file_path.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal denied")

    if size is not None:
        return await _serve_resized(file_path, size, split, class_name, filename)

    if file_path.is_file():
        media_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
        return FileResponse(file_path, media_type=media_type)

    return await _hf_proxy(class_name, filename)


@router.get("/summary")
async def dataset_summary():
    """Summary of the dataset: total species and total images."""
    dataset_dirs = _get_dataset_dirs()
    valid = set(settings.class_names) | {settings.novel_class_name}
    all_species: dict[str, int] = {}
    for d in dataset_dirs:
        species_counts = await run_in_threadpool(_count_species_files, d)
        for name, count in species_counts.items():
            if name in valid:
                all_species[name] = all_species.get(name, 0) + count
    return {"total_species": len(all_species), "total_images": sum(all_species.values())}


@router.get("/gallery")
async def dataset_gallery(user=Depends(require_role("user"))):
    """Flat list of all dataset images across all classes and splits."""
    dataset_dirs = _get_dataset_dirs()
    images: list[dict] = []
    valid = set(settings.class_names) | {settings.novel_class_name}
    for d in dataset_dirs:
        split = d.name
        for species_dir in sorted(d.iterdir()):
            if not species_dir.is_dir() or species_dir.name not in valid:
                continue
            for f in sorted(species_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                    images.append(
                        {
                            "filename": f.name,
                            "split": split,
                            "class_name": species_dir.name,
                            "url": f"/api/dataset/images/{split}/{species_dir.name}/{f.name}",
                        }
                    )
    return images
