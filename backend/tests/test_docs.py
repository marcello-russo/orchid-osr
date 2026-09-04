import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_has_orchid_help():
    txt = _read("README.md")
    assert "orchid --help" in txt or "orchid train --help" in txt
    assert "orchid figures --all" in txt
    assert "§3.5" in txt or "3.5" in txt
    assert "orchid" in txt.lower()
    # deprecated alias still documented
    assert "python cli.py" in txt or "cli.py" in txt


def test_readme_it_has_orchid_help():
    txt = _read("README.it.md")
    assert "orchid" in txt.lower()
    assert "orchid figures --all" in txt


def test_agents_has_orchid_cli():
    txt = _read("AGENTS.md")
    assert "orchid" in txt.lower()
    assert "orchid --help" in txt or "orchid train" in txt
    assert "orchid figures --all" in txt
