def test_exp10_archived():
    import pathlib

    # In CI backend/experiments is empty (gitignored) — check definitions, not filesystem
    from app.cli.definitions import EXPERIMENTS, EXPERIMENTS_ARCHIVED

    assert 10 not in EXPERIMENTS
    assert 10 in EXPERIMENTS_ARCHIVED
    # Filesystem check only if experiments dir exists locally
    exp_path = (
        pathlib.Path("backend/experiments")
        if pathlib.Path("backend/experiments").exists()
        else pathlib.Path("experiments")
    )
    if exp_path.exists():
        assert not list(exp_path.glob("exp10_*"))
    assert (
        "_archived/" in pathlib.Path("backend/.gitignore").read_text()
        if pathlib.Path("backend/.gitignore").exists()
        else pathlib.Path(".gitignore").read_text()
    )
    # README may be at backend/scripts/... or scripts/...
    assert (
        pathlib.Path("backend/scripts/archive/thesis-figures/README.md").exists()
        or pathlib.Path("scripts/archive/thesis-figures/README.md").exists()
    )
