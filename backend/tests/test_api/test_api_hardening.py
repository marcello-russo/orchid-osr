# tests/test_api/test_api_hardening.py
"""Bordi dell'API visti rotti o pericolosi sul campo.

Tre regressioni reali, ognuna osservata in produzione:

- il corpo d'errore riportava ``"request_id": "???"`` (mappa globale indicizzata
  su ``id(request)``, che ``BaseHTTPMiddleware`` invalida);
- un secondo middleware CORS con wildcard rispondeva ``*`` a origini estranee;
- un percorso API o un asset inesistente tornava ``200`` + ``index.html``,
  mascherando i typo dei client.
"""

import os
import re
import time
from email.utils import formatdate, parsedate_to_datetime
from pathlib import Path

import pytest

RID_RE = re.compile(r"^[0-9a-f]{8}$")


class TestRequestId:
    def test_error_body_has_generated_id(self, client):
        # POST su una route solo-GET: 405 dal router, formato dall'handler.
        resp = client.post("/api/experiments")
        assert resp.status_code == 405
        assert RID_RE.match(resp.json()["request_id"]), resp.json()

    def test_header_and_body_agree(self, client):
        resp = client.post("/api/experiments")
        assert resp.headers["x-request-id"] == resp.json()["request_id"]

    def test_client_supplied_id_is_echoed(self, client):
        resp = client.post("/api/experiments", headers={"X-Request-ID": "abc12345"})
        assert resp.json()["request_id"] == "abc12345"
        assert resp.headers["x-request-id"] == "abc12345"

    def test_client_supplied_id_is_sanitized(self, client):
        # L'header finisce nei log: niente spazi, newline o 16 KB di testo.
        resp = client.post("/api/experiments", headers={"X-Request-ID": "a b\t<script>x" * 10})
        rid = resp.headers["x-request-id"]
        assert re.match(r"^[A-Za-z0-9._-]{1,64}$", rid), rid
        assert resp.json()["request_id"] == rid

    def test_unusable_id_falls_back_to_generated(self, client):
        resp = client.post("/api/experiments", headers={"X-Request-ID": "!!!@@@///"})
        assert RID_RE.match(resp.headers["x-request-id"])

    def test_validation_error_carries_id(self, client):
        resp = client.post("/auth/login", json={"username": 1})
        assert resp.status_code == 422
        assert RID_RE.match(resp.json()["request_id"]), resp.json()


class TestCors:
    def test_dev_origin_preflight_allowed(self, client):
        resp = client.options(
            "/api/experiments",
            headers={"Origin": "http://localhost:5175", "Access-Control-Request-Method": "GET"},
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://localhost:5175"

    def test_unknown_origin_gets_no_cors_header(self, client):
        resp = client.get("/api/experiments", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in resp.headers

    def test_single_cors_header(self, client):
        # Due middleware CORS sovrapposti producevano header doppi: i browser
        # rifiutano la risposta con più valori.
        resp = client.get("/api/experiments", headers={"Origin": "http://localhost:5175"})
        assert len(resp.headers.get_list("access-control-allow-origin")) == 1

    def test_origins_are_configurable(self):
        from app.config import Settings

        s = Settings(cors_origins="https://one.example, https://two.example ,")
        assert s.cors_origins_list() == ["https://one.example", "https://two.example"]


class TestSpaFallback:
    # Le route vere della SPA (frontend/src/App.tsx): un refresh su ognuna deve
    # tornare index.html. `/auth` e `/classify` sono i casi che contano, perché
    # condividono il prefisso con l'API — la prima versione di questa regola
    # li mandava in 404 e rompeva il login (il redirect su 401 va a `/auth`).
    @pytest.mark.parametrize(
        "path",
        [
            "/",
            "/auth",
            "/auth/",
            "/classify",
            "/research",
            "/dataset",
            "/experiments",
            "/experiments/exp1",
            "/admin/users",
        ],
    )
    def test_spa_route_serves_index(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert resp.headers["content-type"].startswith("text/html"), path

    @pytest.mark.parametrize("path", ["/api/does-not-exist", "/api", "/auth/nope", "/classify/nope", "/health/nope"])
    def test_api_prefix_typo_is_json_404(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 404, path
        assert resp.headers["content-type"].startswith("application/json"), path
        assert resp.json()["error"] == "http_error"

    def test_api_route_still_wins_on_its_own_path(self, client):
        """`/classify/models` è una GET dell'API: il fallback SPA non la copre."""
        assert client.get("/classify/models").status_code in (200, 401, 403)

    def test_health_stays_json(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")

    def test_missing_asset_is_404_not_html(self, client):
        resp = client.get("/figures/nope.webp")
        assert resp.status_code == 404
        assert not resp.headers["content-type"].startswith("text/html")

    def test_existing_asset_is_served(self, client):
        resp = client.get("/assets/app.js")
        assert resp.status_code == 200
        assert resp.text == "// test"

    def test_head_on_page_is_200(self, client):
        assert client.head("/research").status_code == 200

    @pytest.mark.parametrize("path", ["/", "/research", "/experiments/exp1"])
    def test_index_is_not_cached_without_revalidation(self, client, path):
        """`index.html` deve essere rivalidato: è la mappa degli asset.

        Senza `Cache-Control` il browser usa la freschezza euristica e dopo un
        deploy può chiedere `/assets/index-<hash-vecchio>.js`, che non esiste
        più: pagina bianca senza nessun errore lato server. Il guasto si vede
        solo in produzione (curl non ha cache), quindi il test lo blocca qui.
        """
        resp = client.get(path)
        assert resp.headers.get("cache-control") == "no-cache", path

    def test_index_revalidates_with_etag_instead_of_resending(self, client):
        """`no-cache` non è "scarica sempre": con ETag uguale si risponde 304."""
        first = client.get("/research")
        etag = first.headers["etag"]
        second = client.get("/research", headers={"If-None-Match": etag})
        assert second.status_code == 304
        assert second.content == b""
        # Il 304 deve dire al client di continuare a rivalidare, o la copia
        # salvata tornerebbe "fresca" e non verrebbe più ricontrollata.
        assert second.headers.get("cache-control") == "no-cache"
        assert second.headers["etag"] == etag
        # Ma non `Content-Length`: un 304 non ha corpo, e uvicorn con
        # `--http httptools` solleva `Response content shorter than
        # Content-Length` se l'header c'è (con h11 no, quindi non lo vedrebbe
        # nessun test che non attraversi un trasporto vero).
        assert "content-length" not in second.headers

    def test_index_revalidates_with_last_modified(self, client):
        """`If-Modified-Since` è il validatore che resta dietro Cloudflare.

        Sull'HTML la riva toglie l'ETag (verificato live: `/favicon.svg` lo
        conserva, `/` no), quindi senza questa via il browser non potrebbe mai
        ottenere un 304 in produzione.
        """
        first = client.get("/")
        last_modified = first.headers["last-modified"]
        second = client.get("/", headers={"If-Modified-Since": last_modified})
        assert second.status_code == 304
        # Il 304 deve ripetere la data, o la rivalidazione successiva non ha
        # niente su cui basarsi.
        assert second.headers["last-modified"] == last_modified
        # Una data precedente all'ultima modifica significa "copia vecchia".
        older = formatdate(parsedate_to_datetime(last_modified).timestamp() - 60, usegmt=True)
        assert client.get("/", headers={"If-Modified-Since": older}).status_code == 200

    def test_if_none_match_wins_over_if_modified_since(self, client):
        """RFC 9110 §13.1.3: con `If-None-Match` presente, la data si ignora."""
        resp = client.get(
            "/",
            headers={
                "If-None-Match": '"non-e-questo"',
                "If-Modified-Since": formatdate(time.time() + 60, usegmt=True),
            },
        )
        assert resp.status_code == 200

    def test_malformed_if_modified_since_is_not_a_500(self, client):
        """Un header che nessuno valida non deve trasformare la pagina in un 500."""
        assert client.get("/", headers={"If-Modified-Since": "ieri sera"}).status_code == 200

    @pytest.mark.parametrize("header", ["W/{etag}", '"{altro}", {etag}', "*"])
    def test_index_etag_matches_weak_list_and_star_forms(self, client, header):
        """Un browser può mandare `W/"…"`, una lista o `*`: è la stessa cosa."""
        etag = client.get("/").headers["etag"]
        value = header.format(etag=etag, altro="deadbeef")
        assert client.get("/", headers={"If-None-Match": value}).status_code == 304

    def test_head_revalidates_too(self, client):
        """I monitor usano HEAD: lì il 304 deve funzionare senza corpo."""
        etag = client.get("/").headers["etag"]
        resp = client.head("/research", headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""

    def test_index_with_a_stale_etag_is_served_again(self, client):
        """Indice cambiato dopo il deploy: 200 con il corpo nuovo, non un 304."""
        resp = client.get("/", headers={"If-None-Match": '"non-e-questo"'})
        assert resp.status_code == 200
        assert "test" in resp.text

    def test_a_rebuilt_index_changes_its_etag(self, client):
        """Ricostruire la dist cambia mtime, quindi validatore e risposta."""
        path = Path(os.environ["FRONTEND_DIST"]) / "index.html"
        before = path.stat()
        old_etag = client.get("/").headers["etag"]
        try:
            os.utime(path, (before.st_atime, before.st_mtime + 10))
            resp = client.get("/", headers={"If-None-Match": old_etag})
            assert resp.status_code == 200
            assert resp.headers["etag"] != old_etag
        finally:
            os.utime(path, (before.st_atime, before.st_mtime))

    def test_direct_index_html_has_the_same_rule(self, client):
        """`/index.html` non è normalizzato a `/` da Cloudflare: stessa regola."""
        resp = client.get("/index.html")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "no-cache"

    def test_etag_is_exactly_the_one_starlette_would_send(self):
        """La formula replica `FileResponse` apposta: se cambia, lo dice qui.

        Un bump di Starlette (Dependabot lo apre) che cambia la formula non
        romperebbe niente in modo visibile: semplicemente l'ETag del 200 e
        quello calcolato qui divergerebbero, e nessun client otterrebbe più un
        304. Meglio saperlo da un test che da un `curl` in produzione.
        """
        from starlette.responses import FileResponse

        from app.api.main import _index_etag

        path = Path(os.environ["FRONTEND_DIST"]) / "index.html"
        stat = path.stat()
        assert _index_etag(stat) == FileResponse(path, stat_result=stat).headers["etag"]

    def test_hashed_assets_keep_their_own_caching(self, client):
        """Gli asset hanno l'hash nel nome: la regola dell'indice non li tocca."""
        resp = client.get("/assets/app.js")
        assert resp.headers.get("cache-control") != "no-cache"

    def test_unhandled_error_has_header_and_body_id(self, client):
        """Il 500 è gestito fuori dai middleware: l'header va aggiunto a mano."""
        from fastapi.testclient import TestClient

        from app.api.main import app

        async def _boom():
            raise RuntimeError("boom")

        before = list(app.router.routes)
        app.add_api_route("/__boom", _boom, methods=["GET"], include_in_schema=False)
        app.router.routes.insert(0, app.router.routes.pop())  # prima del catch-all SPA
        try:
            resp = TestClient(app, raise_server_exceptions=False).get("/__boom")
        finally:
            app.router.routes[:] = before
        assert resp.status_code == 500
        assert resp.json()["request_id"] == resp.headers["x-request-id"]
