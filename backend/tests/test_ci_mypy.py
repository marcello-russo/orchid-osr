def test_ci_has_mypy_job():
    import pathlib

    # Find ci.yml relative to repo root (works both locally and in CI)
    candidates = [
        pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml",
        pathlib.Path(".github/workflows/ci.yml"),
        pathlib.Path("../.github/workflows/ci.yml"),
    ]
    ci_path = next((p for p in candidates if p.exists()), candidates[0])
    txt = ci_path.read_text()
    assert "mypy" in txt
    assert "app/cli" in txt
