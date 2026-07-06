"""Admin endpoints: sync DB, health, etc."""

import logging

from fastapi import APIRouter, Depends, Request

from app.api.deps import require_role
from app.api.routes.auth import limiter
from app.core.db import sync_all

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/sync-db")
@limiter.limit("5/minute")
async def sync_database(request: Request, admin=Depends(require_role("admin"))):
    try:
        synced = sync_all()
        return {"synced": synced, "status": "ok"}
    except Exception as e:
        logger.error("DB sync fallita: %s", e)
        return {"synced": 0, "status": "error", "detail": str(e)}
