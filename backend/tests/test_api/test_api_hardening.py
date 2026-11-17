# tests/test_api/test_api_hardening.py
"""Bordi dell'API visti rotti o pericolosi sul campo.

Tre regressioni reali, ognuna osservata in produzione:

- il corpo d'errore riportava ``"request_id": "???"`` (mappa globale indicizzata
  su ``id(request)``, che ``BaseHTTPMiddleware`` invalida);
- un secondo middleware CORS con wildcard rispondeva ``*`` a origini estranee;
- un percorso API o un asset inesistente tornava ``200`` + ``index.html``,
  mascherando i typo dei client.
"""

import re

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
