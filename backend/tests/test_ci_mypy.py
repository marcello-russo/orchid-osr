# tests/test_ci_mypy.py
"""Il gate mypy deve controllare qualcosa.

Il job esisteva da mesi ed era verde: peccato che ogni modulo di `app/cli`
avesse `ignore_errors = true` in pyproject.toml, quindi passavano solo i due
`__init__.py` vuoti. Un `return "non un int"` in `runner.py` non faceva
fallire niente. Questo test legge la configurazione vera e la confronta con
lo scope che la CI passa a mypy.
"""

import fnmatch
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CI = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "backend" / "pyproject.toml"


def _ci_code() -> str:
    return "\n".join(line for line in CI.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#"))


def _mypy_targets() -> list[str]:
    """Percorsi passati a `mypy` nella CI (le opzioni, cioè `-…`, si scartano)."""
    targets: list[str] = []
    for line in _ci_code().splitlines():
        if not re.search(r"\bmypy\s", line):
            continue
        args = line.split("mypy ", 1)[1].split()
        targets += [a.rstrip("/") for a in args if not a.startswith("-")]
    return targets


def _mypy_config() -> dict:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["tool"]["mypy"]


def test_ci_has_a_blocking_mypy_job():
    code = _ci_code()
    assert "mypy" in code
    assert "app/cli" in code
    # Era `continue-on-error: true`: poteva essere rosso per settimane senza
    # fermare un merge.
    assert "continue-on-error" not in code


def test_mypy_targets_are_not_silenced_by_ignore_errors():
    targets = _mypy_targets()
    assert targets, "la CI non lancia mypy su nessun percorso"
    overrides = _mypy_config().get("overrides", [])
    silenced = [
        (target, ov["module"])
        for target in targets
        for ov in overrides
        if ov.get("ignore_errors")
        for module in [target.replace("/", ".")]
        # un override `app.cli.*` copre anche i moduli dentro app/cli
        if fnmatch.fnmatch(module, ov["module"]) or ov["module"].startswith(module + ".")
    ]
    assert not silenced, f"mypy gira su {targets} ma questi override lo azzerano: {silenced}"


def test_mypy_scope_keeps_the_rest_of_strict():
    """L'unica deroga ammessa sullo scope è sulle annotazioni mancanti.

    Se qualcuno rilassa altro (es. `check_untyped_defs = false`, che salta i
    corpi delle funzioni non annotate) il gate torna decorativo.
    """
    cfg = _mypy_config()
    assert cfg["strict"] is True
    relaxed = {"disallow_untyped_defs", "disallow_untyped_calls", "disallow_incomplete_defs"}
    targets = [t.replace("/", ".") for t in _mypy_targets()]
    for ov in cfg.get("overrides", []):
        if not any(t == ov["module"] or fnmatch.fnmatch(t, ov["module"]) for t in targets):
            continue
        assert set(ov) - {"module"} <= relaxed, f"override troppo permissivo: {ov}"
        assert "check_untyped_defs" not in ov, "salterebbe i corpi delle funzioni non annotate"
