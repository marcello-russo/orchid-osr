def test_figures_style_exists():
    from app.figures.style import MATPLOTLIB_RC, PALETTE

    assert len(PALETTE) == 6
    assert PALETTE[0] == "#0173B2"
    assert "figure.dpi" in MATPLOTLIB_RC
    assert MATPLOTLIB_RC["figure.dpi"] == 300


def test_figures_modules_expose_the_builders_the_cli_calls():
    # Prima qui c'era `assert hasattr(m, "build") or ... or True`: sempre vero,
    # quindi non proteggeva niente (e infatti il CLI chiamava `sync_figures`,
    # che non esiste, senza che nessun test se ne accorgesse).
    from app.figures import closed, gcd, osr, reliability, sync

    assert callable(closed.build_closed)
    assert callable(closed.build_heatmap_13)
    assert callable(gcd.build_gcd)
    assert callable(osr.build_osr)
    assert callable(reliability.build_reliability)
    assert callable(sync.sync_frontend)


def test_sync_frontend_copies_png_and_webp(tmp_path):
    from app.figures.sync import sync_frontend

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "fig.png").write_bytes(b"png")
    (src / "fig.webp").write_bytes(b"webp")
    (src / "fig.pdf").write_bytes(b"pdf")

    sync_frontend(src, dst)

    assert sorted(p.name for p in dst.iterdir()) == ["fig.png", "fig.webp"]

