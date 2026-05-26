# backend/app/api/main.py
import logging
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import auth, classify, comparison, dataset, embeddings, experiments, health, species
from app.api.routes import ncd as ncd_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Orchid Classification API",
    description="API per classificare specie di orchidee Ophrys con modelli multi-architettura",
    version="1.0.0",
)


@app.on_event("startup")
async def _startup_sync_weights():
    """Pre-download all model weights from HuggingFace Hub in background."""
    from app.api.deps import download_all_weights
    from app.core.auth import ensure_admin_exists
    ensure_admin_exists()
    def _sync():
        try:
            download_all_weights(background=False)
        except Exception as e:
            logger.error(f"Weight sync failed: {e}")
    thread = threading.Thread(target=_sync, daemon=True)
    thread.start()
    logger.info("Background weight sync started")

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:5175", "http://127.0.0.1:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(health.router, tags=["health"])
app.include_router(classify.router, tags=["classification"])
app.include_router(experiments.router, tags=["experiments"])
app.include_router(species.router, tags=["species"])
app.include_router(dataset.router, tags=["dataset"])
app.include_router(comparison.router, tags=["comparison"])
app.include_router(embeddings.router, tags=["embeddings"])
app.include_router(ncd_router.router, tags=["ncd"])

# Serve frontend static files in production (when frontend/dist exists)
_static_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _static_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=_static_dir / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SPA — return index.html for all non-API, non-asset routes."""
        if ".." in full_path:
            return FileResponse(_static_dir / "index.html")
        file_path = _static_dir / full_path
        try:
            file_path.resolve().relative_to(_static_dir.resolve())
        except ValueError:
            return FileResponse(_static_dir / "index.html")
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_static_dir / "index.html")
