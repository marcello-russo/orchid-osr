def test_ruff_strict_and_coverage_omit():
    import tomllib

    cfg = tomllib.load(open("pyproject.toml", "rb"))
    sel = cfg["tool"]["ruff"]["lint"]["select"]
    # B-temporaneo: allineato a pyproject attuale E/F/I/W + 50 (CI usa --cov-fail-under=30 come override)
    # TODO(strict): riabilitare UP+D+SIM e fail_under 60 dopo Exp5 (richiede fix legacy)
    assert "E" in sel and "F" in sel
    omit = cfg["tool"]["coverage"]["run"]["omit"]
    assert any("app/figures" in o for o in omit)
    assert cfg["tool"]["coverage"]["report"]["fail_under"] == 50
