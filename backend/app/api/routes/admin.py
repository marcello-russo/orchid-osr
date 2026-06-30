"""Admin endpoints: sync DB, health, etc."""

import logging

from fastapi import APIRouter, Depends

from app.api.deps import require_role
from app.core.db import sync_all

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/sync-db")
async def sync_database(admin=Depends(require_role("admin"))):
    synced = sync_all()
    return {"synced": synced, "status": "ok"}
