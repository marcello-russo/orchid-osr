# backend/app/api/main.py
import logging
import mimetypes
import os
import re
import threading
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
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
from app.config import settings

logger = logging.getLogger(__name__)


_RID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _new_request_id(raw: str | None) -> str:
    """Id di correlazione: quello del client se accettabile, altrimenti nuovo.

    Il valore arriva da un header, quindi non si riflette verbatim: senza
    filtro un client può iniettare 16 KB di testo nei log e nell'header di
    risposta (e riusare l'id di un'altra richiesta). Ammessi solo i caratteri
    di un id plausibile, max 64.
    """
    if raw:
        cleaned = _RID_UNSAFE.sub("", raw)[:64]
        if cleaned:
            return cleaned
    return str(uuid.uuid4())[:8]


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Aggiunge un request_id univoco a ogni richiesta per tracciabilità.

    L'id vive in ``request.state`` (che è agganciato allo ``scope``) e non in
    un dizionario globale: ``BaseHTTPMiddleware`` crea un ``Request`` diverso da
    quello che arriva agli handler, quindi una mappa indicizzata su ``id()``
    perdeva il valore e gli errori riportavano ``"???"``.
    """

    async def dispatch(self, request: Request, call_next):
        request.state.request_id = _new_request_id(request.headers.get("X-Request-ID"))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


def _req_id(request: Request) -> str:
    """request_id della richiesta corrente, generandolo se assente.

    Gli handler d'errore (405, 404, 500) possono girare su richieste che non
    hanno attraversato il middleware: meglio un id nuovo che ``"???"``.
    """
    rid = getattr(request.state, "request_id", None)
    if rid:
        return str(rid)
    rid = _new_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = rid
    return rid


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup: sync pesi, DB, periodico. Shutdown: cleanup HTTP client.

    Background threads (weights sync, periodic DB sync) are daemon=True:
    they terminate with the process and do not require explicit shutdown.
    The only resource that needs cleanup is _http_client (aiohttp session).
    """
    from app.api.weights import download_all_weights
    from app.core.auth import ensure_superadmin_exists

    ensure_superadmin_exists()

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
    # Il tag è l'unica fonte di verità della versione: compose.yml lo passa nel
    # container, così /openapi.json dice cosa gira davvero invece di un
    # "1.0.0" hardcoded che non corrispondeva a nessuna release.
    version=os.environ.get("ORCHID_VERSION", "dev"),
    lifespan=lifespan,
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
    # L'header va messo qui: l'handler `Exception` gira in ServerErrorMiddleware,
    # che sta FUORI dai middleware utente, quindi RequestIDMiddleware non vede
    # questa risposta e senza header il caso più importante (500) resta senza
    # id correlabile a monte.
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "Errore interno del server",
            "request_id": rid,
        },
        headers={"X-Request-ID": rid},
    )


# CORS — in produzione la SPA è servita dalla stessa origine, quindi l'elenco
# riguarda solo il dev server Vite. Configurabile da env (CORS_ORIGINS, separati
# da virgola): un solo middleware, nessun wildcard, e credenziali disattivate
# perché l'auth viaggia in `Authorization: Bearer`, non in cookie.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
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

# Serve frontend static files in production (when frontend/dist exists).
# Layout: <repo>/frontend/dist in locale, /workspace/frontend/dist nell'immagine.
# FRONTEND_DIST permette di puntare altrove senza cambiare il codice.
# Python 3.11 non conosce .webp (né .woff2/.map): senza questa registrazione
# `FileResponse` li serve come `application/octet-stream`. I browser li
# mostrano lo stesso, ma cache/CDN e client severi guardano il Content-Type.
for _ext, _mime in (
    (".webp", "image/webp"),
    (".woff2", "font/woff2"),
    (".map", "application/json"),
):
    mimetypes.add_type(_mime, _ext)

_static_dir = Path(os.environ.get("FRONTEND_DIST") or Path(__file__).resolve().parents[3] / "frontend" / "dist")

# Prefissi che appartengono all'API, non alla SPA: un percorso sbagliato qui
# deve dare JSON 404, non `index.html` con stato 200 (mascherava i typo nei
# client e nei monitor, che vedevano "HTML" al posto di un errore).
_RESERVED_PREFIXES = frozenset({"api", "auth", "classify", "health"})

# Due di quei prefissi sono anche route della SPA (`App.tsx`): `/auth` è la
# pagina di login e `/classify` quella di classificazione. La forma esatta è
# della SPA (un refresh o un redirect a sessione scaduta ci finisce sopra),
# tutto ciò che sta sotto (`/auth/me`, `/classify/models`) resta API.
_SPA_EXACT_ROUTES = frozenset({"auth", "classify"})

if _static_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=_static_dir / "assets"), name="assets")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
    async def serve_spa(full_path: str):
        """Serve the SPA — return index.html for all non-API, non-asset routes.

        Anche HEAD: i monitor e i link checker lo usano, e un 405 su `/` fa
        sembrare il sito giù. `FileResponse` gestisce HEAD da sé (header senza
        corpo), quindi qui non serve altro.
        """
        head = full_path.split("/", 1)[0]
        if head in _RESERVED_PREFIXES and full_path.rstrip("/") not in _SPA_EXACT_ROUTES:
            raise HTTPException(status_code=404, detail="Not Found")
        if ".." in full_path:
            return FileResponse(_static_dir / "index.html")
        file_path = _static_dir / full_path
        try:
            file_path.resolve().relative_to(_static_dir.resolve())
        except ValueError:
            return FileResponse(_static_dir / "index.html")
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        # Un percorso che sembra un file (`*.webp`, `*.js`, `*.png`) e non
        # esiste è un 404, non la SPA: prima tornava 200 text/html e un'immagine
        # rotta sembrava caricata.
        if "." in full_path.rsplit("/", 1)[-1]:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(_static_dir / "index.html")
