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


def _module(target: str) -> str:
    """`app/core/auth.py` → `app.core.auth`; `app/cli/` → `app.cli`."""
    return target.rstrip("/").removesuffix(".py").replace("/", ".")


def _matches(module: str, pattern: str) -> bool:
    """fnmatch da solo non copre `app.cli.*` quando il target è `app.cli`."""
    return fnmatch.fnmatch(module, pattern) or pattern == f"{module}.*"


def _patterns(override: dict) -> list[str]:
    """`module` in un override TOML può essere una stringa o una lista."""
    module = override["module"]
    return [module] if isinstance(module, str) else list(module)


def _most_specific(module: str, overrides: list[dict]) -> dict | None:
    """Replica la precedenza di mypy: con più override vince il più specifico.

    Verificato sul campo: con `app.core.*` + `app.core.auth`, l'ordine nel
    pyproject.toml è irrilevante e il secondo riaccende il gate dentro il
    wildcard. Il test non deve quindi accontentarsi di "esiste un override che
    non spegne": deve guardare quello che mypy applicherà davvero.
    """
    matches = [(pattern, ov) for ov in overrides for pattern in _patterns(ov) if _matches(module, pattern)]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]


def test_mypy_targets_are_not_silenced_by_ignore_errors():
    targets = _mypy_targets()
    assert targets, "la CI non lancia mypy su nessun percorso"
    overrides = _mypy_config().get("overrides", [])
    silenced = [
        (module, _patterns(ov))
        for module in (_module(t) for t in targets)
        if (ov := _most_specific(module, overrides)) and ov.get("ignore_errors")
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
    overrides = cfg.get("overrides", [])
    for target in _mypy_targets():
        ov = _most_specific(_module(target), overrides)
        if ov is None:
            continue
        assert set(ov) - {"module", "ignore_errors"} <= relaxed, f"override troppo permissivo: {ov}"
        if "ignore_errors" in ov:
            # `false` è legittimo: riaccende il gate dentro un wildcard.
            assert ov["ignore_errors"] is False, f"override spegne il gate: {ov}"
        assert "check_untyped_defs" not in ov, "salterebbe i corpi delle funzioni non annotate"


def test_typed_scope_includes_the_api_and_auth_storage():
    """Il debito dichiarato non deve tornare a coprire API e storage auth."""
    targets = {_module(t) for t in _mypy_targets()}
    assert "app.api" in targets
    assert "app.preprocessing" in targets
    assert "app.config" in targets
    assert {"app.core.auth", "app.core.auth_store", "app.core.db"} <= targets


def test_mypy_job_installs_the_packages_whose_types_it_checks():
    """Senza fastapi/starlette/typer ogni decoratore vale `Any`.

    Sotto `strict` è `untyped-decorator`: 62 errori che esistono solo sul runner
    (dove si installa mypy e nient'altro) e zero in locale. La tentazione è
    spegnere `disallow_untyped_decorators`; la cura è dare a mypy gli stessi
    pacchetti che ha in locale, alla versione del lock — il lock intero, così
    non esiste un secondo elenco da tenere allineato.
    """
    block = _job_block("mypy")
    install = [line for line in block.splitlines() if "pip install" in line]
    assert any("mypy==" in line for line in install), "il job mypy non installa mypy"
    assert any(
        "-r requirements.txt" in line for line in install
    ), f"il job mypy non installa il lock: gli untyped-decorator tornano. Trovato: {install}"
    assert '"mypy==1.19.1"' in block, "mypy deve essere pinnato, non l'ultima release"
    lock = (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8")
    for package in ("fastapi==", "starlette==", "typer=="):
        assert re.search(rf"^{package}", lock, re.MULTILINE), f"{package[:-2]} non è pinnato nel lock"


def test_mypy_job_installs_the_cpu_torch_first():
    """accelerate/peft/captum tirano dentro torch: senza pin arriva il wheel CUDA.

    Il lock non contiene torch (per scelta, vedi l'intestazione di
    requirements.txt) ma tre sue dipendenze lo richiedono. Sul runner questo
    significava 3,2 GB di wheel `nvidia-*` e una versione di torch scelta da
    PyPI quel giorno — cioè il gate che dipende dalla release di ieri, più il
    rischio di timeout a cache fredda. Il job backend lo evita da sempre; qui
    si verifica che anche mypy lo faccia, e **prima** del lock.
    """
    lines = [line.strip() for line in _job_block("mypy").splitlines() if "pip install" in line]
    torch_line = next((i for i, line in enumerate(lines) if "torch-pins.txt" in line), None)
    assert torch_line is not None, f"il job mypy non installa torch dai pin CPU: {lines}"
    assert "--index-url https://download.pytorch.org/whl/cpu" in lines[torch_line], lines[torch_line]
    lock_line = next(i for i, line in enumerate(lines) if "-r requirements.txt" in line)
    assert torch_line < lock_line, "torch va installato prima del lock, o pip lo risolve da PyPI"
