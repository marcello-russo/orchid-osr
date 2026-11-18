"""Admin endpoints: sync DB, cache management, etc."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import require_role
from app.api.routes.auth import limiter
from app.api.schemas import CacheStatus
from app.api.weights import clear_cache, delete_cached_weight, download_all_weights, get_cache_status
from app.core.db import sync_all

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/sync-db")
@limiter.limit("5/minute")
async def sync_database(request: Request, admin: dict[str, Any] = Depends(require_role("admin"))) -> dict[str, Any]:
    try:
        synced = sync_all()
        return {"synced": synced, "status": "ok"}
    except Exception as e:
        logger.error("DB sync fallita: %s", e)
        return {"synced": 0, "status": "error", "detail": str(e)}


# ── Cache management ────────────────────────────────────────


@router.get("/cache", response_model=CacheStatus)
@limiter.limit("10/minute")
async def cache_status(request: Request, admin: dict[str, Any] = Depends(require_role("admin"))) -> CacheStatus:
    return get_cache_status()


@router.post("/cache/download")
@limiter.limit("2/minute")
async def cache_download_all(
    request: Request, admin: dict[str, Any] = Depends(require_role("admin"))
) -> dict[str, str]:
    download_all_weights(background=True)
    return {"status": "started"}


@router.post("/cache/download/{model_id}")
@limiter.limit("5/minute")
async def cache_download_one(
    model_id: str, request: Request, admin: dict[str, Any] = Depends(require_role("admin"))
) -> dict[str, str]:
    from app.api.weights import _resolve_weights

    result = _resolve_weights(model_id)
    if result is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return {"status": "ok", "model_id": model_id}


@router.delete("/cache/{model_id}")
@limiter.limit("10/minute")
async def cache_delete_one(
    model_id: str, request: Request, admin: dict[str, Any] = Depends(require_role("admin"))
) -> dict[str, str]:
    deleted = delete_cached_weight(model_id)
    if not deleted:
        raise HTTPException(404, f"No cached weights for {model_id}")
    return {"status": "deleted", "model_id": model_id}


@router.delete("/cache")
@limiter.limit("5/minute")
async def cache_clear(request: Request, admin: dict[str, Any] = Depends(require_role("admin"))) -> dict[str, Any]:
    count = clear_cache()
    return {"status": "cleared", "files_removed": count}
