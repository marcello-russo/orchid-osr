import asyncio
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.api.deps import get_current_user_optional
from app.api.schemas import ClassImagesResponse, DatasetClassesResponse, DatasetClassInfo, DatasetImage
from app.config import settings

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VALID_SPLITS = {"train", "test"}
_VALID_CLASSES = set(settings.class_names)
_CROPPED_DIR = "cropped_256x512"
_HF_BRANCH = "main"
_LISTING_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_listing.json"

# Quante immagini per classe vede chi non ha un account. Il limite e' applicato
# qui, non solo nella UI: l'anteprima pubblica deve restare un'anteprima.
PREVIEW_PER_CLASS = 3

_http_client: httpx.AsyncClient | None = None
_listing_cache: dict[str, Any] | None = None
_preview_cache: set[tuple[str, str, str]] | None = None


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


def _get_listing() -> dict[str, Any]:
    global _listing_cache
    if _listing_cache is not None:
        return _listing_cache
    if _LISTING_PATH.is_file():
        with open(_LISTING_PATH) as f:
            listing: Any = json.load(f)
        _listing_cache = listing if isinstance(listing, dict) else None
        if _listing_cache is not None:
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


def _preview_paths() -> set[tuple[str, str, str]]:
    """(split, classe, file) servibili senza account: le prime N per classe."""
    global _preview_cache
    if _preview_cache is not None:
        return _preview_cache
    listing = _get_listing()
    out: set[tuple[str, str, str]] = set()
    for class_name in settings.class_names:
        files = listing.get("files", {}).get("train", {}).get(class_name, [])
        if not files:
            class_dir = _get_dataset_base() / "train" / class_name
            if class_dir.is_dir():
                files = sorted(f.name for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
        for filename in files[:PREVIEW_PER_CLASS]:
            out.add(("train", class_name, filename))
    _preview_cache = out
    return out


@router.get("/classes", response_model=DatasetClassesResponse)
async def list_classes() -> DatasetClassesResponse:
    listing = _get_listing()
    classes: list[DatasetClassInfo] = []
    total_train = 0
    total_test = 0
    for class_name, counts in sorted(listing.get("classes", {}).items()):
        # Solo le 6 specie dello studio: altre entità in listing (es. popolazioni
        # rimosse come O. sphegodes_Napoli) non hanno file servibili e darebbero
        # 548 miniature rotte in galleria.
        if class_name not in settings.class_names:
            continue
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
                    if cn not in settings.class_names:
                        continue
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
    user: dict[str, Any] | None = Depends(get_current_user_optional),
) -> ClassImagesResponse:
    class_name = _validate_segment(class_name)
    if class_name not in settings.class_names:
        raise HTTPException(status_code=404, detail="Classe non presente nel dataset dello studio")
    if split not in _VALID_SPLITS:
        raise HTTPException(status_code=400, detail="Invalid split. Use: train or test")

    if user is None:
        limit = min(limit, PREVIEW_PER_CLASS)
        offset = 0

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


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Scrive il file e poi lo rinomina: nessun lettore vede un file a metà.

    Serve perché più richieste concorrenti sulla stessa immagine (galleria
    dataset) possono scrivere e servire lo stesso percorso. Scrivendo in place,
    la `FileResponse` di un'altra richiesta stima il `Content-Length` su un file
    ancora incompleto e poi legge più byte del dichiarato: il server solleva
    `LocalProtocolError: Too much data for declared Content-Length` e il client
    vede un 520.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


async def _hf_proxy(class_name: str, filename: str, cache_dir: Path | None = None) -> Response:
    """Proxy image from HuggingFace dataset CDN, optionally caching to disk.

    Un'immagine che non esiste in nessun crop è un 404; un errore transitorio
    del CDN (429/5xx) no: quello è un 502, altrimenti la galleria mostra foto
    "mancanti" che invece ci sono.
    """
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    statuses: list[int] = []
    for crop in [
        "cropped/224x224/train",
        "cropped/224x224/test",
        "cropped/256x512/train",
        "cropped/256x512/test",
    ]:
        hf_url = f"https://huggingface.co/datasets/{settings.hf_dataset_repo}/resolve/{_HF_BRANCH}/{crop}/{class_name}/{filename}"
        client = await _get_http_client()
        for attempt in range(2):
            resp = await client.get(hf_url, headers=headers, follow_redirects=True)
            if resp.status_code < 500 and resp.status_code != 429:
                break
            statuses.append(resp.status_code)
            if attempt == 0:
                await asyncio.sleep(0.4)
        statuses.append(resp.status_code)
        if resp.status_code == 200:
            # Cache to disk if requested
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                _write_bytes_atomic(cache_dir / filename, resp.content)
            content_type = resp.headers.get("content-type", "image/jpeg")
            return Response(content=resp.content, media_type=content_type)
    if any(status == 429 or status >= 500 for status in statuses):
        raise HTTPException(status_code=502, detail="HuggingFace CDN non raggiungibile, riprova")
    raise HTTPException(status_code=404, detail="Image not found locally or on HuggingFace")


def _thumbnails_root() -> Path:
    """Radice delle miniature: `THUMBNAILS_DIR` se impostata, altrimenti accanto al dataset.

    In produzione `THUMBNAILS_DIR` punta al volume (`/data/.thumbnails`), così
    le miniature sopravvivono ai redeploy: senza, ogni riavvio del container
    ripartiva con la cache fredda e rifaceva 24 fetch dal CDN per ogni galleria.
    """
    override = os.environ.get("THUMBNAILS_DIR")
    if override:
        return Path(override)
    return _get_dataset_base().parent / ".thumbnails"


async def _serve_resized(file_path: Path, size: int, split: str, class_name: str, filename: str) -> Response:
    """Generate and serve a resized WebP thumbnail, fetching from HF if needed."""
    cache_base = _thumbnails_root() / str(size)
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

    def _generate() -> bool:
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(img_path) as src:
                w, h = src.size
                # `Image.open` restituisce `ImageFile`, `resize` un `Image`:
                # dichiarare il tipo evita di inferire `ImageFile` e poi
                # assegnargli un `Image`.
                img: Image.Image = src
                if w > size:
                    img = src.resize((size, int(h * (size / w))), Image.Resampling.BILINEAR)
                img.save(tmp_path, "WEBP", quality=80)
            # Rinomina atomica: un'altra richiesta che serve lo stesso file
            # non lo vede mai a metà (vedi _write_bytes_atomic).
            os.replace(tmp_path, cache_path)
            return True
        except Exception as e:
            print(f"Thumbnail generation failed: {e}")
            return False
        finally:
            tmp_path.unlink(missing_ok=True)

    success = await run_in_threadpool(_generate)
    if success and cache_path.is_file():
        return FileResponse(cache_path, media_type="image/webp")

    # Fallback: serve the original (locale o appena scaricata) senza resize
    for candidate in (file_path, img_path):
        if candidate.is_file():
            media_type = mimetypes.guess_type(str(candidate))[0] or "image/jpeg"
            return FileResponse(candidate, media_type=media_type)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/{split}/{class_name}/{filename}")
async def serve_image(
    split: str,
    class_name: str,
    filename: str,
    size: int | None = Query(None, gt=0, le=1024),
    user: dict[str, Any] | None = Depends(get_current_user_optional),
) -> Response:
    """Serve a dataset image file, optionally resized and converted to WebP.

    Checks local filesystem first, then falls back to HuggingFace CDN proxy.
    Images fetched from CDN are cached locally for subsequent requests.
    """
    split = _validate_segment(split)
    class_name = _validate_segment(class_name)
    filename = _validate_segment(filename)
    if user is None and (split, class_name, filename) not in _preview_paths():
        raise HTTPException(
            status_code=403,
            detail="Anteprima pubblica: accedi per vedere la galleria completa",
        )
    # Gli embedding usano nomi split NCD (es. test_ncd_exp9): normalizza
    if split.startswith("test"):
        split = "test"
    elif split.startswith("train"):
        split = "train"
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
