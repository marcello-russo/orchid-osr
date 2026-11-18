#!/usr/bin/env python3
"""Smoke live: verifica l'immagine *deployed*, non il codice nel repo.

Serve perche' un 200 non dice niente: la pagina Dataset risponde 200 anche
quando nessuna immagine arriva, `/health` risponde 200 anche se il CDN e' giu'
e un errore mascherato e' spesso un 200 con content-type `application/json`.
Qui si controllano i byte (magic bytes, content-type, lunghezza), i totali del
dataset e che la versione in `/openapi.json` sia quella che si e' appena
deployata: se il container non e' stato ricreato, si vede.

Uso:
    scripts/smoke-live.py                      # https://osr.marrusso.dev
    scripts/smoke-live.py --url http://127.0.0.1:8000 --env-file /srv/orchid/secrets/app.env
    scripts/smoke-live.py --expect-version v0.4.2

Le credenziali (superadmin) si leggono dal file di env indicato, o da
ORCHID_SMOKE_USER/ORCHID_SMOKE_PASSWORD; senza, i controlli autenticati vengono
saltati (e lo dice). Non stampa mai token ne' password.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get("ORCHID_URL", "https://osr.marrusso.dev")
DEFAULT_ENV_FILE = os.environ.get("ORCHID_SMOKE_ENV", "/srv/orchid/secrets/app.env")

# Un'immagine e' un'immagine solo se comincia cosi'. Un errore JSON o una pagina
# HTML possono arrivare con 200: il content-type da solo non basta.
MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",
    b"GIF8": "gif",
    b"BM": "bmp",
}

failures: list[str] = []
skipped: list[str] = []


def read_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def request(url: str, token: str | None = None, body: dict | None = None) -> tuple[int, str, bytes]:
    headers = {"Accept": "*/*", "User-Agent": "orchid-smoke/1"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.headers.get("content-type", ""), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("content-type", ""), exc.read()
    except urllib.error.URLError as exc:
        return 0, "", str(exc.reason).encode()


def check(label: str, path: str, token: str | None = None, body: dict | None = None) -> tuple[int, str, bytes]:
    status, ctype, payload = request(BASE + path, token, body)
    if status >= 400 or status == 0:
        failures.append(f"{label}: {status} {payload[:200]!r}")
        print(f"ERR {status:3} {len(payload):8} B  {label:34} {path}")
    else:
        print(f"ok  {status:3} {len(payload):8} B  {label:34} {path}")
    return status, ctype, payload


def check_image(label: str, path: str, token: str, expect: tuple[str, ...] = ()) -> None:
    status, ctype, payload = check(label, path, token)
    if status >= 400:
        return
    kind = next((name for magic, name in MAGIC.items() if payload.startswith(magic)), None)
    if kind is None:
        failures.append(f"{label}: non e' un'immagine (magic {payload[:8]!r}, content-type {ctype})")
        print(f"    ^ non e' un'immagine: {ctype} {payload[:40]!r}")
        return
    if expect and kind not in expect:
        failures.append(f"{label}: formato {kind}, atteso {expect}")
    if len(payload) < 1024:
        failures.append(f"{label}: {len(payload)} byte, troppo pochi per un'immagine")
    print(f"    ^ {kind}, {ctype}, {len(payload)} B")


def main() -> int:
    global BASE
    parser = argparse.ArgumentParser(description="Smoke di un deploy OrchID (API, immagini, SPA).")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"base URL (default {DEFAULT_URL})")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="file con SUPERADMIN_USERNAME/PASSWORD")
    parser.add_argument("--expect-version", default="", help="versione attesa in /openapi.json (es. v0.4.2)")
    parser.add_argument("--min-images", type=int, default=2000, help="immagini minime nel dataset")
    parser.add_argument("--min-experiments", type=int, default=1, help="esperimenti minimi")
    args = parser.parse_args()

    BASE = args.url.rstrip("/")
    print(f"==> smoke {BASE}")

    # --- l'immagine che gira e' quella che si e' deployata? -----------------
    status, _, payload = check("openapi", "/openapi.json")
    version = ""
    if status < 400:
        try:
            info = json.loads(payload)["info"]
            version = info.get("version", "")
            print(f"    versione in esecuzione: {version}")
        except (json.JSONDecodeError, KeyError):
            failures.append("openapi: risposta senza info.version")
    if args.expect_version and version != args.expect_version:
        failures.append(f"versione: gira {version!r}, attesa {args.expect_version!r} (container non ricreato?)")

    # --- SPA ---------------------------------------------------------------
    status, ctype, payload = check("SPA", "/")
    if status < 400:
        if "text/html" not in ctype:
            failures.append(f"SPA: content-type {ctype}")
        if b'id="root"' not in payload:
            failures.append('SPA: manca <div id="root">, la dist non e\' quella giusta')

    # --- dataset: elenco, file, byte ---------------------------------------
    status, _, payload = check("specie (totali)", "/api/species")
    if status < 400:
        data = json.loads(payload)
        print(
            f"    {data.get('total_species')} specie, {data.get('total_images')} immagini, "
            f"train+test come da split pulito"
        )
        if int(data.get("total_images", 0)) < args.min_images:
            failures.append(f"dataset: {data.get('total_images')} immagini, attese almeno {args.min_images}")

    token = None
    username = os.environ.get("ORCHID_SMOKE_USER", "")
    password = os.environ.get("ORCHID_SMOKE_PASSWORD", "")
    if not (username and password) and Path(args.env_file).is_file():
        env = read_env_file(args.env_file)
        username = env.get("SUPERADMIN_USERNAME", "")
        password = env.get("SUPERADMIN_PASSWORD", "")
    if username and password:
        status, _, payload = check("login", "/auth/login", body={"username": username, "password": password})
        if status < 400:
            token = json.loads(payload)["token"]
            role = json.loads(payload)["user"]["role"]
            print(f"    ruolo: {role}")
            if role != "superadmin":
                failures.append(f"login: ruolo {role!r}, atteso superadmin")
    else:
        skipped.append("login e controlli autenticati: nessuna credenziale")
        print("SKIP login/immagini: nessuna credenziale")

    if token:
        status, _, payload = check("classi del dataset", "/api/dataset/classes", token)
        classes = []
        if status < 400:
            classes = json.loads(payload)["classes"]
            names = [c["name"] for c in classes]
            print(f"    {len(classes)} specie, train={sum(c['train_count'] for c in classes)}")
            if not classes:
                failures.append("classi: la pagina Dataset sarebbe vuota (listing assente?)")
            if len(names) != len(set(names)):
                failures.append(f"classi: duplicati {names}")

        if classes:
            first = classes[0]["name"]
            quoted = urllib.parse.quote(first)
            path = f"/api/dataset/classes/{quoted}/images?split=train&limit=2"
            status, _, payload = check("file di una classe", path, token)
            files = json.loads(payload)["images"] if status < 400 else []
            if not files:
                failures.append(f"immagini: {first} non elenca nessun file")
            else:
                name = urllib.parse.quote(files[0]["filename"])
                check_image("immagine dal CDN", f"/api/dataset/images/train/{quoted}/{name}", token)
                thumb = f"/api/dataset/images/train/{quoted}/{name}?size=224"
                check_image("miniatura 224 (WebP)", thumb, token, expect=("webp",))

        # --- esperimenti (le pagine della tesi) ----------------------------
        status, _, payload = check("elenco esperimenti", "/api/experiments", token)
        if status < 400:
            data = json.loads(payload)
            # L'endpoint risponde con una lista; il serializer storicamente
            # poteva incapsularla in {"experiments": [...]}. Accetta entrambe.
            items = data["experiments"] if isinstance(data, dict) else data
            print(f"    {len(items)} esperimenti")
            if len(items) < args.min_experiments:
                failures.append(f"esperimenti: {len(items)}, attesi almeno {args.min_experiments}")
            if items:
                exp_id = items[0].get("id") or items[0].get("name")
                check("dettaglio esperimento", f"/api/experiments/{urllib.parse.quote(str(exp_id))}", token)
        check("matrice benchmark", "/api/comparison/experiments", token)

    print()
    if skipped:
        for note in skipped:
            print(f"SKIP {note}")
    if failures:
        print("FALLIMENTI:")
        for item in failures:
            print(f" - {item}")
        return 1
    print("smoke OK: nessuna risposta >= 400, immagini decodificabili, versione coerente")
    return 0


BASE = ""

if __name__ == "__main__":
    sys.exit(main())
