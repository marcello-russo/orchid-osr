"""Il README pubblico deve documentare la CLI vera."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_has_orchid_help():
    txt = _read("README.md")
    assert "orchid --help" in txt
    assert "orchid train" in txt
    assert "orchid figures --all" in txt
    # l'alias deprecato resta documentato
    assert "python cli.py" in txt


def test_readme_it_has_orchid_help():
    txt = _read("README.it.md")
    assert "orchid --help" in txt
    assert "orchid train" in txt
    assert "orchid figures --all" in txt
