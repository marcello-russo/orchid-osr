# tests/test_quality_gate.py
"""Il gate di qualità è codice: se cambia, un test deve accorgersene.

Le regole vivono in `pyproject.toml` (unica fonte) e in `.github/workflows/ci.yml`.
Averle in due posti le aveva già fatte divergere (soglia 30 in CI, 50 nel
progetto, nessuno se ne accorgeva). Questo test legge i file veri.
"""

import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _cfg() -> dict:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


def test_ruff_rules_are_strict_without_dead_ignores():
    lint = _cfg()["tool"]["ruff"]["lint"]
    assert {"E", "F", "I", "W"} <= set(lint["select"])
    # S = regole di sicurezza (bandit). Se qualcuno le toglie per far passare
    # build, questo test lo dice.
    assert "S" in lint["select"]
    # Gli ignore per-file devono puntare a qualcosa che esiste: in passato
    # contenevano preprocessing_tools.py e tests/test_integration/*, cioè glob
    # morti che nascondevano debito senza che nessuno se ne accorgesse.
    backend = PYPROJECT.parent
    for pattern in lint.get("per-file-ignores", {}):
        assert list(backend.glob(pattern)), f"per-file-ignores su un path inesistente: {pattern}"


def test_coverage_measures_the_api_and_has_one_threshold():
    cov = _cfg()["tool"]["coverage"]
    assert cov["run"]["branch"] is True
    # L'API era esclusa dalla misura: la route più grande non aveva test.
    assert cov["run"]["omit"] == ["app/figures/*"]
    # Soglia 3 punti sotto il misurato (51,7%): regressioni vere passano, codice
    # di training non ancora testabile non fa fallire la build per un decimale.
    assert cov["report"]["fail_under"] == 48


def _ci_code() -> str:
    """Solo le righe eseguibili: i commenti citano le opzioni per spiegarle."""
    return "\n".join(line for line in CI.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#"))


def test_ci_does_not_override_the_coverage_threshold():
    ci = _ci_code()
    assert "--cov-fail-under" not in ci, "la soglia deve stare solo in pyproject.toml"
    assert "pytest" in ci and "--cov=app" in ci


def test_ci_runs_the_frontend_tests_and_i18n_check():
    ci = CI.read_text(encoding="utf-8")
    # 110 test vitest esistevano e nessuno li eseguiva.
    assert "npm test" in ci
    assert "check-i18n" in ci


def test_ci_installs_the_same_pins_as_the_image():
    ci = _ci_code()
    # torch/torchvision: un solo file di pin per Dockerfile e CI.
    assert "-r torch-pins.txt" in ci
    assert "-r requirements-dev.txt" in ci
    # mypy non è più decorativo.
    assert "continue-on-error" not in ci


def _req_names(path: pathlib.Path) -> set[str]:
    """Nomi dei pacchetti di un requirements file (normalizzati, senza pin)."""
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(_package_name(line))
    return names


def _package_name(spec: str) -> str:
    return re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].lower().replace("_", "-")


def test_lock_covers_every_declared_dependency():
    """Il lock è generato: se si aggiunge una dipendenza e non lo si rigenera,
    l'immagine non la installa e il guasto compare solo a runtime.

    Non si rigenera qui (`uv pip compile` dipende da rete e versione di uv):
    si controlla che ogni nome dichiarato in pyproject.toml esista nel lock
    giusto. La rigenerazione resta `bash scripts/ci/sync_requirements.sh`.
    """
    backend = PYPROJECT.parent
    project = _cfg()["project"]
    runtime = _req_names(backend / "requirements.txt")
    dev = _req_names(backend / "requirements-dev.txt")
    torch_pins = _req_names(backend / "torch-pins.txt")

    declared = {_package_name(d) for d in project["dependencies"]}
    missing = declared - runtime - dev - torch_pins
    assert not missing, f"dipendenze in pyproject.toml ma non nel lock: {sorted(missing)}"

    declared_dev = {_package_name(d) for d in project["optional-dependencies"]["dev"]}
    missing_dev = declared_dev - dev
    assert not missing_dev, f"dev extra non in requirements-dev.txt: {sorted(missing_dev)}"
