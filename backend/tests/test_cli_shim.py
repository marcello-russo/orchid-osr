import pathlib


def test_cli_shim_deprecated():
    txt = pathlib.Path("cli.py").read_text()
    assert "DeprecationWarning" in txt
    assert "orchid train" in txt
    assert 'if __name__ == "__main__"' in txt
