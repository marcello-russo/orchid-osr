import pathlib


def test_closed_builds_png(tmp_path: pathlib.Path):
    from app.figures.closed import build_closed

    out = build_closed(tmp_path, {})
    assert len(out) >= 2
    assert all(p.suffix in (".png", ".webp", ".pdf") for p in out)
    assert all(p.stat().st_size > 1000 for p in out)


def test_osr_builds(tmp_path: pathlib.Path):
    from app.figures.osr import build_osr

    out = build_osr(tmp_path, {})
    assert len(out) >= 2
    assert all(p.suffix in (".png", ".webp", ".pdf") for p in out)
    assert all(p.stat().st_size > 1000 for p in out)


def test_gcd_builds(tmp_path: pathlib.Path):
    from app.figures.gcd import build_gcd

    out = build_gcd(tmp_path, {})
    # build_gcd returns a list of Paths (png/webp/pdf)
    assert isinstance(out, list) and len(out) >= 2
    assert all(p.suffix in (".png", ".webp", ".pdf") for p in out)
    assert all(p.stat().st_size > 1000 for p in out)


def test_reliability_builds(tmp_path: pathlib.Path):
    from app.figures.reliability import build_reliability

    out = build_reliability(tmp_path, {})
    assert len(out) >= 2
    assert all(p.suffix in (".png", ".webp", ".pdf") for p in out)
    assert all(p.stat().st_size > 1000 for p in out)


def test_heatmap_13_builds_png(tmp_path):
    from app.figures.closed import build_heatmap_13

    paths = build_heatmap_13(tmp_path, {})
    names = [p.name for p in paths]
    assert "heatmap_1_3.png" in names
    assert (tmp_path / "heatmap_1_3.png").stat().st_size > 10000
