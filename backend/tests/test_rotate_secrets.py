# tests/test_rotate_secrets.py
"""La rotazione di un segreto è un'operazione che si sbaglia una volta su tre.

Storia vera: per capire come era fatto il container ho stampato l'ambiente e
`HF_TOKEN`, `JWT_SECRET` e `SUPERADMIN_PASSWORD` sono finiti in un transcript.
Ruotarli a mano significa riscrivere un file che è la fonte di verità del
servizio: una riga duplicata, un file troncato o un `mode` cambiato non si
vedono fino al deploy successivo. Questi test usano il file vero (in una
directory temporanea) e controllano il contenuto, non solo il codice di uscita.
"""

import importlib.util
import re
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rotate-app-secrets.py"

ORIGINAL = """\
# Produzione — segreti (mode 600)
SUPERADMIN_USERNAME=admin
SUPERADMIN_PASSWORD=vecchia-password
JWT_SECRET=vecchio-jwt
HF_DATASET_REPO=marcellorusso/orchid-osr-dataset
HF_MODELS_REPO=marcellorusso/orchid-osr-models
HF_TOKEN=hf_tokenvecchio

# commento finale
"""


def _load():
    spec = importlib.util.spec_from_file_location("rotate_app_secrets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "app.env"
    path.write_text(ORIGINAL, encoding="utf-8")
    path.chmod(0o600)
    return path


def _values(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            result[key] = value
    return result


def test_rotates_both_and_leaves_everything_else_untouched(env_file: Path):
    module = _load()
    assert module.main(["--env-file", str(env_file)]) == 0

    values = _values(env_file)
    original = {line.split("=", 1)[0]: line.split("=", 1)[1] for line in ORIGINAL.splitlines() if "=" in line}
    assert values["JWT_SECRET"] != original["JWT_SECRET"]
    assert values["SUPERADMIN_PASSWORD"] != original["SUPERADMIN_PASSWORD"]
    for key in ("SUPERADMIN_USERNAME", "HF_DATASET_REPO", "HF_MODELS_REPO", "HF_TOKEN"):
        assert values[key] == original[key], f"{key} non doveva cambiare"

    text = env_file.read_text(encoding="utf-8")
    assert text.startswith("# Produzione — segreti (mode 600)\n"), "il commento di testa è sparito"
    assert text.endswith("# commento finale\n"), "il commento di coda è sparito"
    assert text.count("JWT_SECRET=") == 1, "chiave duplicata: vince l'ultima, in silenzio"
    assert text.count("SUPERADMIN_PASSWORD=") == 1


def test_new_values_have_the_expected_shape(env_file: Path):
    module = _load()
    assert module.main(["--env-file", str(env_file), "--rotate", "jwt"]) == 0
    jwt = _values(env_file)["JWT_SECRET"]
    assert re.fullmatch(r"[0-9a-f]{64}", jwt), f"JWT_SECRET non è 32 byte hex: {jwt[:4]}…"
    # HS256 con meno di 256 bit di entropia è una chiave debole, non un segreto.
    assert len(set(jwt)) > 8


def test_password_is_alphanumeric_and_long_enough(env_file: Path):
    module = _load()
    assert module.main(["--env-file", str(env_file), "--rotate", "superadmin"]) == 0
    password = _values(env_file)["SUPERADMIN_PASSWORD"]
    assert re.fullmatch(r"[A-Za-z0-9]{24}", password)
    assert password != "vecchia-password"


def test_two_runs_give_different_secrets(env_file: Path):
    module = _load()
    module.main(["--env-file", str(env_file)])
    first = _values(env_file)
    module.main(["--env-file", str(env_file)])
    second = _values(env_file)
    assert first["JWT_SECRET"] != second["JWT_SECRET"]
    assert first["SUPERADMIN_PASSWORD"] != second["SUPERADMIN_PASSWORD"]


def test_mode_is_preserved(env_file: Path):
    module = _load()
    module.main(["--env-file", str(env_file)])
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600, "un segreto leggibile da group/other"


def test_never_prints_the_values(env_file: Path, capsys):
    module = _load()
    assert module.main(["--env-file", str(env_file)]) == 0
    out = capsys.readouterr()
    values = _values(env_file)
    assert values["JWT_SECRET"] not in out.out + out.err
    assert values["SUPERADMIN_PASSWORD"] not in out.out + out.err
    assert "hunter2" not in out.out


def test_dry_run_does_not_write(env_file: Path):
    module = _load()
    before = env_file.read_bytes()
    assert module.main(["--env-file", str(env_file), "--dry-run"]) == 0
    assert env_file.read_bytes() == before


def test_hf_token_is_refused_with_a_reason(env_file: Path, capsys):
    module = _load()
    before = env_file.read_bytes()
    assert module.main(["--env-file", str(env_file), "--rotate", "hf"]) == 2
    assert env_file.read_bytes() == before
    err = capsys.readouterr().err
    assert "HuggingFace" in err, "il rifiuto deve dire dove si crea il token"
    assert "write" in err, "il rifiuto deve dire che scope serve"


def test_a_duplicated_key_is_an_error_not_a_silent_overwrite(env_file: Path, capsys):
    env_file.write_text(ORIGINAL + "JWT_SECRET=secondo-valore\n", encoding="utf-8")
    module = _load()
    before = env_file.read_bytes()
    assert module.main(["--env-file", str(env_file), "--rotate", "jwt"]) == 2
    assert env_file.read_bytes() == before, "ha toccato un file con una chiave duplicata"
    assert "compare 2 volte" in capsys.readouterr().err


def test_missing_file_is_an_error(tmp_path: Path):
    module = _load()
    assert module.main(["--env-file", str(tmp_path / "nope.env")]) == 2


def test_unknown_target_is_an_error(env_file: Path):
    module = _load()
    assert module.main(["--env-file", str(env_file), "--rotate", "database"]) == 2


def test_the_new_password_reaches_the_database_on_restart():
    """Ruotare il file senza riavviare lascia il DB con l'hash vecchio: il
    vecchio login continuerebbe a funzionare. L'allineamento avviene all'avvio."""
    compose = (ROOT / "compose.yml").read_text(encoding="utf-8")
    assert "secrets/app.env" in compose, "il compose non passa il file dei segreti al container"

    main_py = (ROOT / "backend" / "app" / "api" / "main.py").read_text(encoding="utf-8")
    assert "ensure_superadmin_exists()" in main_py, "nessun passaggio che allinei l'hash del superadmin"
    assert "lifespan" in main_py or "startup" in main_py, "l'allineamento non avviene all'avvio"
