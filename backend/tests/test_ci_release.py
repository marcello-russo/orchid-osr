# tests/test_ci_release.py
"""La catena tag → build → Release → deploy deve restare una sola catena.

Publish e deploy sono due caller di ~20 righe che delegano al dispatcher
centrale (`marcello-russo/actions`): qui non si costruisce nulla, si manda un
segnale. Il legame fragile è nei payload: se `draft_release` tornasse al
default, la Release resterebbe in bozza e **il deploy non partirebbe mai**
(l'evento `release: published` non arriverebbe); se il deploy perdesse
`assets`, l'host resterebbe sul compose vecchio e il tag non sarebbe più la
fonte di verità. Questo test legge i workflow veri, non una copia.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _flat(name: str) -> str:
    """Il file senza commenti: i commenti non devono soddisfare le asserzioni."""
    lines = (WORKFLOWS / name).read_text(encoding="utf-8").splitlines()
    return " ".join(line for line in lines if not line.lstrip().startswith("#"))


def test_publish_publishes_the_release_so_the_deploy_can_start():
    publish = _flat("docker-publish.yml")
    assert "event_type=docker-publish" in publish
    assert (
        "client_payload[draft_release]=false" in publish
    ), "con la Release in bozza l'evento `release: published` non arriva e il deploy non parte"
    assert re.search(
        r'client_payload\[services\]=.\[\{"service":"app"', publish
    ), "il payload deve dichiarare il servizio app e i path che lo ricompilano"


def test_deploy_starts_from_the_published_release():
    deploy = _flat("deploy-vps.yml")
    assert re.search(r"release:\s+types: \[published\]", deploy), "il deploy deve partire dalla Release pubblicata"
    assert "event_type=vps-deploy" in deploy
    for field in (
        "compose_dir]=/srv/orchid",
        "compose_file]=compose.yml",
        "version_var]=ORCHID_VERSION",
        "assets]='compose.yml:compose.yml edge.yml:edge.yml'",
        "health_url]=https://osr.marrusso.dev/health",
    ):
        assert field in deploy, f"payload di deploy incompleto: {field}"


def test_the_tag_is_the_only_version_source():
    """Mai un `latest` o un tag hardcoded nel compose di produzione."""
    compose = (ROOT / "compose.yml").read_text(encoding="utf-8")
    assert "ghcr.io/marcello-russo/orchid-osr-app:${ORCHID_VERSION:" in compose
    assert ":latest" not in compose
