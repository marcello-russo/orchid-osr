from typer.testing import CliRunner

from app.cli.main import app


def test_orchid_help():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "train" in r.output
    assert "figures" in r.output


def test_train_help():
    import re

    r = CliRunner().invoke(app, ["train", "--help"])
    assert r.exit_code == 0
    clean = re.sub(r"\x1b\[[0-9;]*m", "", r.output)
    assert "--exp" in clean
