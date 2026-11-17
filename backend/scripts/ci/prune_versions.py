"""Prune delle versioni GHCR: tiene le ultime N versioni taggate (semver).

Eseguito dal workflow `Publish Docker images` dopo il push dell'immagine.
La scelta di *cosa* cancellare è una funzione pura (`plan`) perché è la parte
che può fare danni: la logica sta qui, testata in
`backend/tests/test_ci_prune_versions.py`.

Regole (le stesse di `personale/actions`, più il caso del tag ripubblicato):

- il keep si conta sui **tag versione** `vX.Y.Z` ordinati per semver, non sui
  record del pacchetto: ripubblicare lo stesso tag crea un record nuovo e
  contando i record il prune teneva due volte l'ultimo tag, buttando via un
  rollback;
- un record che resta il più recente per un tag tenuto non si tocca mai;
- un record **duplicato** (tag già spostato su un record più nuovo) si cancella;
- versioni **untagged** mai: sono manifest figli dell'indice (buildx pusha
  indice + figli come record separati), cancellarli = tag che esiste ma
  manifest 404 al pull;
- record con soli tag `sha-<commit>` mai: non sono versioni pubblicate.

Uso:
    GH_TOKEN=... OWNER=squidslab PACKAGE=orchid-app KEEP=3 python3 prune_versions.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

VER = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


def semkey(tag: str) -> tuple[int, int, int]:
    m = VER.fullmatch(tag)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def _tags(version: dict) -> list[str]:
    return list(version.get("metadata", {}).get("container", {}).get("tags") or [])


def plan(versions: list[dict], keep: int) -> tuple[dict[int, str], set[str]]:
    """Decide cosa eliminare.

    Ritorna `(da_eliminare, tag_tenuti)` dove `da_eliminare` mappa
    `version_id -> motivo`.
    """
    tagged = [v for v in versions if _tags(v)]
    for v in tagged:
        v["_vtags"] = [t for t in _tags(v) if VER.fullmatch(t)]

    all_tags = sorted({t for v in tagged for t in v["_vtags"]}, key=semkey, reverse=True)
    keep_tags = set(all_tags[:keep])

    # Ultimo record che porta ciascun tag versione (il "vivo").
    newest: dict[str, int] = {}
    for v in sorted(tagged, key=lambda v: v["created_at"]):
        for t in v["_vtags"]:
            newest[t] = v["id"]

    to_delete: dict[int, str] = {}
    for v in sorted(tagged, key=lambda v: v["created_at"]):
        if not v["_vtags"]:
            continue  # solo sha-<commit>: non è una versione
        stale = any(newest[t] != v["id"] for t in v["_vtags"])
        outside = all(t not in keep_tags for t in v["_vtags"])
        if not (stale or outside):
            continue
        to_delete[v["id"]] = "duplicato di un tag ripubblicato" if stale else f"fuori dalle ultime {keep}"
    return to_delete, keep_tags


def main() -> int:
    token = os.environ["GH_TOKEN"]
    owner, pkg = os.environ["OWNER"], os.environ["PACKAGE"]
    keep = int(os.environ.get("KEEP", "3"))
    base = f"/orgs/{owner}/packages/container/{pkg}"

    def api(method: str, path: str):
        req = urllib.request.Request(
            f"https://api.github.com{path}",
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(req) as r:
                body = r.read().decode()
                return json.loads(body) if body else True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    versions = api("GET", f"{base}/versions?per_page=100") or []
    to_delete, keep_tags = plan(versions, keep)
    by_id = {v["id"]: v for v in versions}

    # Dal più vecchio: se un delete fallisce per permessi ci si ferma senza
    # lasciare il registro in ordine casuale.
    deleted = 0
    for vid in sorted(to_delete, key=lambda i: by_id[i]["created_at"]):
        tags = ",".join(_tags(by_id[vid]))
        try:
            api("DELETE", f"{base}/versions/{vid}")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                print(
                    f"::warning::prune non permesso ({e.code}) su [{tags}]: "
                    "serve admin del pacchetto; il deploy non è toccato"
                )
                break
            raise
        deleted += 1
        print(f"eliminata versione [{tags}] del {by_id[vid]['created_at']} ({to_delete[vid]})")

    kept = ", ".join(sorted(keep_tags, key=semkey)) or "(nessuno)"
    print(
        f"tenuti i tag: {kept} — {deleted} versioni eliminate, "
        f"{len(versions) - len([v for v in versions if _tags(v)])} untagged preservate"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
