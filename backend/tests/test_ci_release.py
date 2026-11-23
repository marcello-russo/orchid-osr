# tests/test_ci_release.py
"""Il job della release deve ripubblicare una bozza, non saltarla.

`git push :refs/tags/vX` seguito dal tag nuovo sulla commit corretta è il modo
normale di correggere un rilascio, ma GitHub converte la release orfana in
**bozza**. Il job diceva "release già presente: nessuna modifica" e usciva: la
versione restava fuori da "Latest", con le note del commit sbagliato, e nessuno
se ne accorgeva (è successo con la v0.4.0). Questo test legge il workflow vero.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
PUBLISH = ROOT / ".github" / "workflows" / "docker-publish.yml"


def _release_step() -> str:
    """Il blocco `run: |` dello step che crea la release (senza commenti)."""
    text = PUBLISH.read_text(encoding="utf-8")
    start = text.index("GitHub Release (note dai commit)")
    end = text.index("prune:", start)
    block = text[start:end]
    return "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))


def test_a_draft_release_is_republished_not_skipped():
    step = _release_step()
    assert "isDraft" in step, "il job non distingue una bozza da una release pubblicata"
    assert re.search(r"gh release edit \"\$TAG\" --draft=false", step), "manca la ripubblicazione della bozza"
    # La guardia "esci subito" deve essere condizionata alla non-bozza: se
    # tornasse incondizionata, il caso bozza resterebbe muto.
    guard = re.search(r'if gh release view "\$TAG".*?exit 0', step, re.DOTALL)
    assert guard is not None, "manca la guardia sulla release già presente"
    assert "--json isDraft" in guard.group(0), "la guardia esce anche sulle bozze"


def test_notes_come_from_the_commits_between_tags():
    step = _release_step()
    assert "--no-merges" in step, "le note devono elencare i commit, non il merge"
    assert "git describe --tags --abbrev=0" in step, "la base del diff è il tag antenato, non il semver più alto"
