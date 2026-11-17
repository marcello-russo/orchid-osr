# tests/test_quality_gate.py
"""Il gate di qualità è codice: se cambia, un test deve accorgersene.

Le regole vivono in `pyproject.toml` (unica fonte) e in `.github/workflows/ci.yml`.
Averle in due posti le aveva già fatte divergere (soglia 30 in CI, 50 nel
progetto, nessuno se ne accorgeva). Questo test legge i file veri.
"""

import pathlib
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
    # Gli ignore per-file puntavano anche a file e cartelle inesistenti
    # (preprocessing_tools.py, tests/test_integration/*): o si pulisce il codice
    # o si dichiara il debito, non si nasconde con un glob morto.
    assert "per-file-ignores" not in lint


def test_coverage_measures_the_api_and_has_one_threshold():
    cov = _cfg()["tool"]["coverage"]
    assert cov["run"]["branch"] is True
    # L'API era esclusa dalla misura: la route più grande non aveva test.
    assert cov["run"]["omit"] == ["app/figures/*"]
    # Soglia 3 punti sotto il misurato (49%): regressioni vere passano, codice
    # di training non ancora testabile non fa fallire la build per un decimale.
    assert cov["report"]["fail_under"] == 46


def _ci_code() -> str:
    """Solo le righe eseguibili: i commenti citano le opzioni per spiegarle."""
    return "\n".join(
        line for line in CI.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")
    )


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
