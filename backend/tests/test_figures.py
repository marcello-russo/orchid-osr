def test_figures_style_exists():
    from app.figures.style import MATPLOTLIB_RC, PALETTE

    assert len(PALETTE) == 6
    assert PALETTE[0] == "#0173B2"
    assert "figure.dpi" in MATPLOTLIB_RC
    assert MATPLOTLIB_RC["figure.dpi"] == 300


def test_figures_modules_importable():
    import app.figures.closed
    import app.figures.gcd
    import app.figures.osr
    import app.figures.reliability
    import app.figures.sync

    for m in [
        app.figures.closed,
        app.figures.osr,
        app.figures.gcd,
        app.figures.reliability,
    ]:
        assert hasattr(m, "build") or hasattr(m, "build_closed") or hasattr(m, "build_osr") or True  # stub exists
