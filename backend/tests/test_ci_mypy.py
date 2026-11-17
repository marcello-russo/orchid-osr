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
    """Percorsi passati a `mypy` nella CI (le opzioni, cioè `-…`, si scartano).

    Si guarda solo la riga che *esegue* mypy (`- run: mypy …`): il nome del
    comando compare anche nei commenti e nel nome dello step, e prenderli per
    argomenti farebbe passare per target parole come "le".
    """
    targets: list[str] = []
    for line in _ci_code().splitlines():
        match = re.match(r"\s*(?:-\s*run:\s*)?mypy\s+(.*)", line)
        if not match:
            continue
        targets += [a.rstrip("/") for a in match.group(1).split() if not a.startswith("-")]
    return targets


def _job_block(name: str) -> str:
    """Il testo di un job di ci.yml (dal nome fino al job successivo)."""
    lines = CI.read_text(encoding="utf-8").splitlines()
    collected: list[str] = []
    inside = False
    for line in lines:
        if re.fullmatch(rf"  {re.escape(name)}:", line):
            inside = True
            continue
        if inside and re.fullmatch(r"  [\w-]+:", line):
            break
        if inside:
            collected.append(line)
    assert collected, f"job {name} non trovato in ci.yml"
    return "\n".join(collected)


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


def test_mypy_job_installs_the_typed_decorators_it_checks():
    """Senza typer installato, `@app.command()` vale `Any`.

    Sotto `strict` questo è `untyped-decorator`: 5 errori che esistono solo sul
    runner (dove si installa mypy e nient'altro), non in locale. La tentazione
    è spegnere `disallow_untyped_decorators`; la cura è invece dare a mypy lo
    stesso pacchetto che ha in locale, alla versione del lock — altrimenti il
    gate è verde o rosso a seconda di cosa è installato, cioè non è un gate.
    """
    block = _job_block("mypy")
    install = [line for line in block.splitlines() if "pip install" in line and "mypy==" in line]
    assert install, "il job mypy non installa mypy"
    assert any(
        '"typer==$TYPER"' in line for line in install
    ), f"il job mypy non installa typer: gli errori untyped-decorator tornano. Trovato: {install}"
    # La versione deve venire dal lock, non da PyPI al momento della run.
    assert "requirements.txt" in block
    assert "s/^typer==//p" in block
    lock = (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^typer==", lock, re.MULTILINE), "typer non è pinnato nel lock"
