# tests/test_smoke_live.py
"""Il deploy deve provare che le immagini arrivano, non solo che il servizio è su.

`/health` risponde 200 anche con la dist sbagliata, con il CDN delle immagini
giù o con il container vecchio: il 3/11 un controllo "tutto verde" è stato
letto da due endpoint che *scansionavano il disco* (vuoto in produzione),
mentre la pagina vera funzionava. Questo test lega le due cose che rendono il
deploy onesto:

1. `scripts/smoke-live.py` guarda i **byte** (magic bytes), i totali del
   dataset e la versione in `/openapi.json` — non solo gli status;
2. `scripts/deploy-vps.sh` lo esegue **dentro** il deploy e fallisce se lo
   smoke fallisce (altrimenti è un controllo che si può dimenticare).

Il test legge gli script veri: un commento non lo fa passare.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "smoke-live.py"
DEPLOY = ROOT / "scripts" / "deploy-vps.sh"


def _smoke() -> str:
    return SMOKE.read_text(encoding="utf-8")


def _deploy() -> str:
    return DEPLOY.read_text(encoding="utf-8")


def test_smoke_checks_bytes_not_only_statuses():
    text = _smoke()
    assert "MAGIC" in text, "lo smoke non guarda i magic bytes: un 200 con JSON passerebbe per 'ok'"
    for magic in ("\\xff\\xd8\\xff", "\\x89PNG", "RIFF"):
        assert magic in text, f"manca il magic bytes {magic}"
    assert 'expect=("webp",)' in text, "la miniatura non è verificata come WebP (è l'unico ramo che passa da Pillow)"
    if re.search(r"status < 400", text):
        assert "non e' un'immagine" in text, "un corpo che non è un'immagine deve essere un fallimento, non un ok"


def test_smoke_verifies_the_version_actually_running():
    text = _smoke()
    assert 'info.get("version"' in text or '["info"]' in text, "lo smoke non legge info.version"
    assert "--expect-version" in text, "manca il confronto con il tag atteso (container non ricreato)"
    assert "container non ricreato" in text, "il messaggio deve dire cosa significa la versione sbagliata"


def test_smoke_checks_the_dataset_the_page_needs():
    """La pagina Dataset vive del listing: se sparisce, /health non se ne accorge."""
    text = _smoke()
    assert "/api/dataset/classes" in text, "lo smoke non verifica l'elenco classi"
    assert "la pagina Dataset sarebbe vuota" in text, "un elenco vuoto deve fallire"


def test_smoke_never_prints_credentials():
    text = _smoke()
    for line in text.splitlines():
        if "print(" not in line:
            continue
        assert "{password}" not in line and "{token}" not in line, f"riga che stampa un segreto: {line.strip()}"
    assert "Non stampa mai token ne' password" in text, "il docstring dovrebbe dichiarare la promessa"


def test_deploy_runs_the_smoke_and_fails_the_deploy():
    text = _deploy()
    assert "smoke-live.py" in text, "il deploy non esegue lo smoke"
    assert '--expect-version "$TAG"' in text, "lo smoke non è legato al tag che si sta deployando"
    run = text.index("python3 bin/smoke-live.py")
    # Deve girare dopo pull/up: prima verificherebbe la versione ancora vecchia.
    assert text.index("docker compose up -d") < run, "lo smoke gira prima del deploy"
    assert re.search(r"smoke live fallito", text[run:]), "il fallimento dello smoke non viene annunciato"
    assert re.search(
        r'if \[ "\$smoke" = "failed" \]; then\s*\n\s*exit 1', text[run:]
    ), "uno smoke fallito deve interrompere il deploy"
    assert re.search(r"rollback: scripts/deploy-vps\.sh", text[run:]), "il messaggio deve dire come tornare indietro"


def test_registry_says_what_runs_and_whether_it_was_verified():
    """`.deployed` senza esito è una promessa non mantenuta: prima diceva "questo
    gira" e basta, e uno smoke fallito lasciava il file identico a un successo."""
    text = _deploy()
    run = text.index("python3 bin/smoke-live.py")
    pending = text.index("write_deployed pending")
    final = text.index('write_deployed "$smoke"')
    # `pending` prima dello smoke: se il deploy viene interrotto a metà verifica,
    # il registro dice comunque cosa sta girando.
    assert pending < run < final, "il registro non racconta né prima né dopo la verifica"
    assert '"smoke":"%s"' in text, "il registro non riporta l'esito della verifica"
    assert "write_deployed pending" in text, "nessuna scrittura `pending` prima dello smoke"
    for outcome in ("ok", "failed", "skipped"):
        assert f'smoke="{outcome}"' in text, f"esito {outcome} non prodotto"
    assert re.search(
        r'write_deployed "\$smoke"\s*\n\s*if \[ "\$smoke" = "failed" \]; then\s*\n\s*exit 1', text
    ), "un deploy non verificato non deve uscire 0"


def test_an_old_tag_without_the_smoke_says_so_instead_of_being_silent():
    """Il rollback a un tag vecchio non contiene lo smoke: deve avvisare, non tacere."""
    text = _deploy()
    staged = r"if \[ -f \"?\$STAGE/scripts/smoke-live\.py\"? \]; then"
    assert re.search(staged, text), "lo stage assume che lo smoke ci sia"
    warned = r"else\s*\n\s*echo \"!! smoke live assente"
    assert re.search(warned, text), "manca l'avviso quando il tag non ha lo smoke"
    assert re.search(warned + " in bin/", text), "l'host tace quando lo smoke non è arrivato"
