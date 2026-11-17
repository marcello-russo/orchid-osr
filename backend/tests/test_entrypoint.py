"""L'entrypoint è PID 1: tre cose che nessun test unitario può eseguire.

Si può lanciare Docker dentro pytest, ma non su runner senza il volume giusto
e senza root: quindi si legge lo script, come `test_ci_mypy.py` legge il
workflow. Ognuna delle righe controllate copre un guasto vero, già visto:

- segnali non inoltrati → `docker stop` aspetta i 30s di grace, poi SIGKILL
  (exit 137) e le richieste in volo vengono troncate;
- DB nel layer scrivibile → i tentativi di login (lockout anti brute-force) si
  azzerano a ogni redeploy;
- `ln -sfn` su una directory già presente → nasce `dataset/dataset`, il volume
  resta vuoto e i download finiscono nel layer dell'immagine.
"""

import pathlib
import re

ENTRYPOINT = pathlib.Path(__file__).resolve().parents[2] / "entrypoint.sh"


def _script() -> str:
    return ENTRYPOINT.read_text(encoding="utf-8")


def _code() -> str:
    """Le righe senza commenti: i commenti parlano dei guasti, non li causano."""
    return "\n".join(line for line in _script().splitlines() if not line.lstrip().startswith("#"))


def test_sigterm_reaches_the_server():
    code = _code()
    assert re.search(r"^trap \w+ TERM INT$", code, re.MULTILINE), "nessun trap su SIGTERM"
    assert 'kill -TERM "$SERVER_PID"' in code
    assert re.search(r'^wait "\$SERVER_PID"$', code, re.MULTILINE), "l'attesa finale deve essere sul figlio"


def test_database_lives_on_the_volume():
    assert (
        'export DB_PATH="$DATA_DIR/orchid.db"' in _code()
    ), "senza DB_PATH il DB resta nel layer scrivibile: il lockout si azzera a ogni redeploy"


def test_dataset_and_experiments_are_linked_without_clobbering():
    code = _code()
    assert "link_to_data dataset" in code
    assert "link_to_data experiments" in code
    # `ln -sfn` su una directory esistente crea il link *dentro*: mai usarlo.
    assert "ln -sfn" not in code
    # La directory esistente si sostituisce solo se è vuota.
    assert 'rmdir "$name" && ln -s' in code
    assert '[ -z "$(ls -A "$name")" ]' in code
