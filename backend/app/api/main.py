# backend/app/api/main.py
import logging
import threading
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import admin as admin_router
from app.api.routes import auth, classify, comparison, dataset, embeddings, experiments, health, species
from app.api.routes import ncd as ncd_router
from app.api.routes.auth import limiter

logger = logging.getLogger(__name__)

_REQUEST_IDS: dict[int, str] = {}


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Aggiunge un request_id univoco a ogni richiesta per tracciabilità."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        _REQUEST_IDS[id(request)] = rid
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            _REQUEST_IDS.pop(id(request), None)


def _req_id(request: Request) -> str:
    return _REQUEST_IDS.get(id(request), "???")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup: sync pesi, DB, periodico. Shutdown: cleanup HTTP client.

    Background threads (weights sync, periodic DB sync) are daemon=True:
    they terminate with the process and do not require explicit shutdown.
    The only resource that needs cleanup is _http_client (aiohttp session).
    """
    from app.api.weights import download_all_weights
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

    # Lazy DB sync (se DB vuoto, popola da JSON in background)
    try:
        from app.core.db import get_db, sync_all

        conn = get_db()
        count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        conn.close()
        if count == 0:
            logger.info("DB vuoto — sync da JSON in background...")
            thread = threading.Thread(target=sync_all, daemon=True)
            thread.start()
        else:
            logger.info("DB ha gia' %d esperimenti — skip sync", count)
    except Exception as e:
        logger.warning("Lazy sync fallita (non bloccante): %s", e)

    # Periodic DB sync ogni 60 secondi (rileva nuovi esperimenti/fold completati)
    def _periodic_sync():
        while True:
            try:
                from app.core.db import sync_all

                n = sync_all()
                if n > 0:
                    logger.info("Sync periodico DB: %d esperimenti aggiornati", n)
            except Exception as e:
                logger.debug("Sync periodico DB: %s", e)
            threading.Event().wait(60)

    thread = threading.Thread(target=_periodic_sync, daemon=True)
    thread.start()
    logger.info("Periodic DB sync started (every 60s)")

    yield

    from app.api.routes.dataset import _http_client

    if _http_client is not None:
        await _http_client.aclose()


app = FastAPI(
    title="Orchid Classification API",
    description="API per classificare specie di orchidee Ophrys con modelli multi-architettura",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    rid = _req_id(request)
    logger.warning("req=%s validation_error: %s", rid, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": "Invalid request data",
            "request_id": rid,
            "fields": {e["loc"][-1]: e["msg"] for e in exc.errors()},
        },
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request, exc):
    rid = _req_id(request)
    if exc.status_code >= 500:
        logger.error("req=%s http_%d: %s", rid, exc.status_code, exc.detail)
    else:
        logger.info("req=%s http_%d: %s", rid, exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail, "request_id": rid},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    rid = _req_id(request)
    logger.error("req=%s unhandled: %s", rid, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "Errore interno del server",
            "request_id": rid,
        },
    )


# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5175", "http://127.0.0.1:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
app.add_middleware(RequestIDMiddleware)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include routers
app.include_router(auth.router)
app.include_router(admin_router.router)
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
