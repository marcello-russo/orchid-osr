# backend/app/api/main.py
import hashlib
import logging
import mimetypes
import os
import re
import threading
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from email.utils import formatdate, mktime_tz, parsedate_tz
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
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

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
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
    from app.core.auth import ensure_superadmin_exists, initialize_auth

    initialize_auth()
    ensure_superadmin_exists()

    def _sync() -> None:
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
    def _periodic_sync() -> None:
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
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
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
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
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
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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


def _handle_rate_limit(request: Request, exc: Exception) -> Response:
    """Adatta l'handler di slowapi alla firma `Exception` di Starlette.

    `add_exception_handler` pretende `Callable[[Request, Exception], Response]`;
    slowapi dichiara il tipo concreto `RateLimitExceeded`, quindi passarlo
    direttamente è un errore di tipo (e un `# type: ignore` qui nasconderebbe
    un cambio di firma vero).
    """
    assert isinstance(exc, RateLimitExceeded)
    return _rate_limit_exceeded_handler(request, exc)


app.add_exception_handler(RateLimitExceeded, _handle_rate_limit)

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


def _index_etag(stat: os.stat_result) -> str:
    """Stesso ETag che darebbe `FileResponse`, calcolato da mtime e dimensione.

    Si replica la formula di Starlette (`set_stat_headers`) apposta: se il
    validatore cambiasse, il browser rimanderebbe l'ETag salvato e non
    otterrebbe mai un 304.
    """
    base = f"{stat.st_mtime}-{stat.st_size}"
    return f'"{hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()}"'


def _etag_matches(header: str | None, etag: str) -> bool:
    """`If-None-Match` è una lista (`W/"a", "b"`), e per una GET il confronto è weak."""
    if not header:
        return False
    if header.strip() == "*":
        return True
    return any(candidate.strip().removeprefix("W/") == etag for candidate in header.split(","))


def _not_modified_since(header: str | None, mtime: float) -> bool:
    """`If-Modified-Since` da solo (RFC 9110 §13.1.2.2).

    Non è ridondante con l'ETag: Cloudflare, sull'HTML, **toglie** l'`ETag` e
    lascia `Last-Modified` (verificato in produzione: `/favicon.svg` conserva
    il suo ETag, `/` no — la pipeline HTML della zona riscrive il corpo, quindi
    un validatore strong non sarebbe più vero). Dietro la riva l'unico
    validatore che il browser può rimandare è quindi la data.

    `mktime_tz` e non `parsedate_to_datetime(...).timestamp()`: il secondo, su
    un datetime naive prodotto da un fuso `-0000`, applica il fuso locale.
    """
    if not header:
        return False
    parsed = parsedate_tz(header)
    if parsed is None:  # header malformato: si risponde 200, non 500
        return False
    return int(mtime) <= mktime_tz(parsed)


def _spa_index(request: Request) -> Response:
    """`index.html` con `Cache-Control: no-cache` e rivalidazione vera.

    È il file che dice al browser *quali* asset caricare (i nomi hanno un hash
    nel contenuto: `index-DBHQmV5M.js`). Senza istruzioni di cache il browser
    applica la freschezza euristica (10% dell'età di Last-Modified) e dopo un
    deploy può tenere l'indice vecchio, che punta a `/assets/index-<hash>.js`
    ormai inesistenti: pagina bianca finché non si svuota la cache, e nessun
    errore lato server che lo spieghi.

    `no-cache` non significa "non salvare": obbliga a rivalidare prima di usare
    la copia. La rivalidazione qui è esplicita perché `FileResponse` da solo
    ignora le precondizioni (le gestisce `StaticFiles`, che però serve le
    directory e non questa route): senza, ogni navigazione riscaricherebbe
    l'indice invece di ricevere un `304` senza corpo.

    Si accettano entrambi i validatori perché a monte c'è una CDN che non
    lascia passare sempre lo stesso: l'ETag quando c'è, altrimenti la data.

    Gli asset sotto `/assets` sono un caso diverso: il nome cambia col
    contenuto, quindi possono stare in cache a lungo — ed è Cloudflare, non
    questo server, a dar loro `max-age`.
    """
    path = _static_dir / "index.html"
    try:
        stat = path.stat()
    except OSError:
        # `dist` sparita dopo l'avvio (il mount è deciso all'import).
        raise HTTPException(status_code=404, detail="Not Found") from None
    etag = _index_etag(stat)
    headers = {
        "Cache-Control": "no-cache",
        "ETag": etag,
        # SHOULD sul 304 (RFC 9110 §15.4.5) e, dietro Cloudflare, l'unico
        # validatore che sopravvive: va ripetuto, o il client non può
        # rivalidare la volta dopo.
        "Last-Modified": formatdate(stat.st_mtime, usegmt=True),
    }
    inm = request.headers.get("if-none-match")
    # RFC 9110 §13.1.3: se c'è `If-None-Match`, `If-Modified-Since` si ignora.
    not_modified = (
        _etag_matches(inm, etag)
        if inm is not None
        else _not_modified_since(request.headers.get("if-modified-since"), stat.st_mtime)
    )
    if not_modified:
        # Niente `Content-Length`: un 304 non ha corpo, e uvicorn con
        # `--http httptools` solleva `Response content shorter than
        # Content-Length` se lo vede (verificato: con h11 no, quindi il guasto
        # arriverebbe solo cambiando trasporto). Starlette lo salta già per i
        # 204/304.
        return Response(status_code=304, headers=headers)
    return FileResponse(path, headers=headers, stat_result=stat)


if _static_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=_static_dir / "assets"), name="assets")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
    async def serve_spa(full_path: str, request: Request) -> Response:
        """Serve the SPA — return index.html for all non-API, non-asset routes.

        Anche HEAD: i monitor e i link checker lo usano, e un 405 su `/` fa
        sembrare il sito giù. `FileResponse` gestisce HEAD da sé (header senza
        corpo), quindi qui non serve altro.
        """
        head = full_path.split("/", 1)[0]
        if head in _RESERVED_PREFIXES and full_path.rstrip("/") not in _SPA_EXACT_ROUTES:
            raise HTTPException(status_code=404, detail="Not Found")
        if ".." in full_path:
            return _spa_index(request)
        file_path = _static_dir / full_path
        try:
            file_path.resolve().relative_to(_static_dir.resolve())
        except ValueError:
            return _spa_index(request)
        if full_path and file_path.is_file():
            # `/index.html` diretto è lo stesso file, quindi la stessa regola:
            # Cloudflare non normalizza quell'URL a `/` (verificato: 200), e un
            # bookmark lì sopra tornerebbe senza `no-cache` — cioè con lo
            # stesso guasto che questa funzione evita.
            if file_path.resolve() == (_static_dir / "index.html").resolve():
                return _spa_index(request)
            return FileResponse(file_path)
        # Un percorso che sembra un file (`*.webp`, `*.js`, `*.png`) e non
        # esiste è un 404, non la SPA: prima tornava 200 text/html e un'immagine
        # rotta sembrava caricata.
        if "." in full_path.rsplit("/", 1)[-1]:
            raise HTTPException(status_code=404, detail="Not Found")
        return _spa_index(request)
