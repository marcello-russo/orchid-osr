#!/usr/bin/env python3
"""Smoke delle *pagine* di produzione (Playwright), non degli endpoint.

Perché esiste, e perché `smoke-live.py` non basta: il 18/11 ho dedotto che la
pagina Dataset fosse vuota da `/api/dataset/summary` e `/api/dataset/gallery`,
due endpoint che scandiscono il disco e che **nessuno chiama**. La pagina
funzionava: 34 `<img>` nel DOM, 34 con `naturalWidth > 0`. La lezione è che
"endpoint vuoto" non è "schermata vuota", e che un `<img>` nel DOM non è
un'immagine: serve `img.complete && img.naturalWidth > 0`.

Qui si aprono le pagine vere in un browser vero, si semina il token nello store
(localStorage) e si verificano le cose che un utente vedrebbe: sezioni del
capitolo, grafici renderizzati, tabella utenti, immagini decodificate. Si
fallisce anche su un solo errore di console o una richiesta fallita (i
`ERR_ABORTED` di navigazione sono rumore e vengono filtrati).

Uso:

    scripts/smoke-pages.py                                  # https://osr.marrusso.dev
    scripts/smoke-pages.py --url http://127.0.0.1:8000 --env-file /srv/orchid/secrets/app.env
    ORCHID_SMOKE_CHROME=/path/to/chrome scripts/smoke-pages.py

Serve Playwright con Chromium (`pip install playwright && playwright install
chromium`): non è nelle dipendenze del progetto perché non gira in CI — è un
controllo manuale post-deploy, e lo dice chiaramente se manca.

Le credenziali arrivano dal file di env (default: quello di produzione) e non
vengono mai stampate: il token vive solo dentro il browser.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get("ORCHID_URL", "https://osr.marrusso.dev")
DEFAULT_ENV_FILE = os.environ.get("ORCHID_SMOKE_ENV", "/srv/orchid/secrets/app.env")
DETAIL_EXPERIMENT = "exp4_sphor_ncd_dinov2_base"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
SECTIONS = (
    "intro",
    "contesto",
    "dataset",
    "metodologia",
    "exp1",
    "exp2",
    "exp3",
    "exp4",
    "exp5",
    "benchmark",
    "conclusioni",
    "bibliografia",
)
INVITE_STATES = ("Disponibile", "Usato", "Scaduto", "Revocato")

problems: list[str] = []


def read_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def login(base: str, username: str, password: str) -> dict:
    """Login via API: la stessa strada della UI, senza passare dal browser."""
    body = json.dumps({"username": username, "password": password}).encode()
    request = urllib.request.Request(
        base + "/auth/login",
        data=body,
        # La WAF della zona (Cloudflare) blocca lo User-Agent di urllib: senza
        # questo header il login da qui è un 403, non un errore di credenziali.
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode("utf-8", "replace")
        raise SystemExit(
            f"login fallito: HTTP {exc.code} {detail}\n"
            "  un 403 da un IP non residenziale è la WAF, non la password: mint il token\n"
            "  sull'host (curl in loopback) e passa --token-file, oppure esegui da lì."
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"login fallito: {exc.reason}") from exc
    if "token" not in payload or "user" not in payload:
        raise SystemExit("login: risposta senza token/user")
    return payload


def run(base: str, session: dict, chrome: str | None) -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**({"executable_path": chrome} if chrome else {}))
        page = browser.new_page(viewport={"width": 1440, "height": 1100})

        def on_console(msg):
            if msg.type in ("error", "warning") and "React DevTools" not in msg.text:
                problems.append(f"console[{msg.type}] {page.url}: {msg.text}")

        def on_failed(request):
            # ERR_ABORTED = navigazione interrotta da noi (o font gstatic): rumore.
            if request.failure and "ERR_ABORTED" in request.failure:
                return
            problems.append(f"requestfailed {page.url}: {request.url} {request.failure}")

        page.on("console", on_console)
        page.on("pageerror", lambda exc: problems.append(f"pageerror {page.url}: {exc}"))
        page.on("requestfailed", on_failed)

        page.goto(f"{base}/auth", wait_until="networkidle")
        page.evaluate(
            """([token, user]) => localStorage.setItem("orchid-app-store", JSON.stringify({
                state: { token, user, lang: "it" }, version: 0,
            }))""",
            [session["token"], session["user"]],
        )

        page.goto(f"{base}/research", wait_until="networkidle")
        page.wait_for_timeout(800)
        for section in SECTIONS:
            if page.locator(f"#{section}").count() != 1:
                problems.append(f"research: sezione #{section} mancante o duplicata")
        toc = page.locator('aside nav a[href^="#"]').count()
        print(f"research: {len(SECTIONS)} sezioni, {toc} voci TOC")
        if toc < len(SECTIONS):
            problems.append(f"research: {toc} voci TOC per {len(SECTIONS)} sezioni")

        page.goto(f"{base}/experiments", wait_until="networkidle")
        page.wait_for_timeout(1500)
        if page.get_by_text("Studio II-A").count() < 1:
            problems.append("experiments: matrice benchmark vuota")
        print("experiments: matrice benchmark popolata")

        page.goto(f"{base}/experiments/{DETAIL_EXPERIMENT}", wait_until="networkidle")
        page.wait_for_timeout(1800)
        if page.get_by_text("Recipe di Training").count() < 1:
            problems.append("detail: card ricetta assente")
        page.locator("button", has_text="Grafici").click()
        page.wait_for_timeout(1500)
        charts = page.locator("svg.recharts-surface").count()
        print(f"detail: {charts} grafici")
        if charts < 1:
            problems.append("detail: nessun grafico renderizzato")
        log_tab = page.locator("button", has_text="Log")
        if log_tab.count() == 1:
            log_tab.click()
            page.wait_for_timeout(800)
            print("detail: tab log aperta")
        else:
            problems.append("detail: tab log non trovata")

        page.goto(f"{base}/admin/users", wait_until="networkidle")
        page.wait_for_timeout(1500)
        rows = page.locator("table tbody tr").count()
        states = {s: page.get_by_text(s).count() for s in INVITE_STATES}
        print(f"admin: {rows} righe, stati {states}")
        if rows < 1:
            problems.append("admin: tabella utenti vuota")
        if sum(states.values()) < 1:
            problems.append("admin: nessuno stato invito")
        if page.get_by_text("Scade il").count() < 1:
            problems.append("admin: colonna scadenza assente")

        # Il controllo che mi ha fatto sbagliare: contano le immagini decodificate.
        page.goto(f"{base}/dataset", wait_until="networkidle")
        page.wait_for_timeout(5000)
        images = page.locator("img")
        total = images.count()
        decoded = [i for i in range(total) if images.nth(i).evaluate("e => e.complete && e.naturalWidth > 0")]
        print(f"dataset: {total} img nel DOM, {len(decoded)} decodificate")
        if total < 20:
            problems.append(f"dataset: solo {total} immagini nel DOM")
        if len(decoded) != total:
            problems.append(f"dataset: {total - len(decoded)} immagini non decodificate")

        page.goto(f"{base}/classify", wait_until="networkidle")
        page.wait_for_timeout(1000)
        print("classify: pagina caricata")

        browser.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke delle pagine (Playwright).")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"base URL (default {DEFAULT_URL})")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="file con SUPERADMIN_USERNAME/PASSWORD")
    parser.add_argument("--chrome", default=os.environ.get("ORCHID_SMOKE_CHROME", ""), help="percorso del Chromium")
    args = parser.parse_args(argv)

    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        print(
            "Playwright non installato: `pip install playwright && playwright install chromium`\n"
            "oppure usa scripts/smoke-live.py (API e byte delle immagini, senza browser).",
            file=sys.stderr,
        )
        return 2

    base = args.url.rstrip("/")
    env = read_env_file(args.env_file)
    username = os.environ.get("ORCHID_SMOKE_USER") or env.get("SUPERADMIN_USERNAME", "")
    password = os.environ.get("ORCHID_SMOKE_PASSWORD") or env.get("SUPERADMIN_PASSWORD", "")
    if not (username and password):
        print(f"errore: nessuna credenziale in {args.env_file} né in ORCHID_SMOKE_USER/PASSWORD", file=sys.stderr)
        return 2

    print(f"==> smoke pagine {base}")
    session = login(base, username, password)
    print(f"    login: ruolo {session['user'].get('role')}")
    run(base, session, args.chrome or None)

    if problems:
        print("\nPROBLEMI:")
        for problem in problems:
            print(f" - {problem}")
        return 1
    print("\nnessun errore console / richiesta fallita")
    return 0


if __name__ == "__main__":
    sys.exit(main())
