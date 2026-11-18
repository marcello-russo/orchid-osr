"""Migrazione legacy e avvio: un `users.json` rotto deve fermare il container."""

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]


def test_startup_fails_loudly_on_malformed_legacy_json(tmp_path):
    """Il lifespan chiama l'import: se il file è corrotto il processo non parte.

    Meglio un container che non parte (con traceback) di un sito che autentica
    contro un DB vuoto perché l'import è stato saltato in silenzio.
    """
    bad = tmp_path / "users.json"
    bad.write_text("{ non-json", encoding="utf-8")

    env = {
        **os.environ,
        "DB_PATH": str(tmp_path / "orchid.db"),
        "USERS_PATH": str(bad),
        "SUPERADMIN_USERNAME": "",
        "SUPERADMIN_PASSWORD": "",
    }
    code = "from fastapi.testclient import TestClient\nfrom app.api.main import app\nwith TestClient(app):\n    pass\n"
    proc = subprocess.run(  # noqa: S603 — eseguiamo il nostro codice, non input utente
        [sys.executable, "-c", code],
        env=env,
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "JSONDecodeError" in proc.stderr or "Expecting property name" in proc.stderr


def test_initialize_auth_is_idempotent(auth_store_db):
    from app.core import auth as auth_module
    from app.core import auth_store

    auth_module.initialize_auth()
    auth_module.initialize_auth()
    assert auth_store.list_users() == []


def test_initialize_auth_reports_the_marker(auth_store_db):
    from app.core import auth as auth_module
    from app.core import auth_store

    auth_module.initialize_auth()
    with auth_store._connection() as conn:
        row = conn.execute("SELECT value FROM auth_meta WHERE key = 'legacy_json_imported'").fetchone()
    assert row is not None
    assert row[0] == "absent"
