# backend/app/api/main.py
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import health, classify, experiments

app = FastAPI(
    title="Orchid Classification API",
    description="API per classificare specie di orchidee Ophrys con ResNet",
    version="1.0.0",
)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(classify.router, tags=["classification"])
app.include_router(experiments.router, tags=["experiments"])

# Serve frontend static files in production (when frontend/dist exists)
_static_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _static_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=_static_dir / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SPA — return index.html for all non-API, non-asset routes."""
        file_path = _static_dir / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_static_dir / "index.html")
