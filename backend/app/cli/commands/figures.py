from __future__ import annotations

import typer


def figures(
    all: bool = typer.Option(True, help="Tutte le 6 figure"),
    only: list[str] | None = typer.Option(None, help="gcd,osr,closed,reliability"),
    check: bool = typer.Option(False, help="Regressione phash"),
    sync_frontend: bool = typer.Option(False, help="Copia in frontend/public/figures"),
) -> None:
    """Generate figures (pixel-perfect, 300dpi) Exp 1-4 + Exp5 extension."""
    from pathlib import Path

    from app.cli.definitions import PROJECT_ROOT

    fig_dir = PROJECT_ROOT.parent / "documents" / "_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Exps mapping 1-5 super researcher — usa FALLBACK se manca, glob per esistenti
    import glob as _glob

    exps: dict[int, Path] = {}
    for i in [1, 2, 3, 4, 5]:
        cands = _glob.glob(str(PROJECT_ROOT / f"experiments/exp{i}_*"))
        if cands:
            # prendi il primo con results.json
            best = next((Path(c) for c in cands if (Path(c) / "results.json").exists()), Path(cands[0]))
            exps[i] = best
        else:
            exps[i] = PROJECT_ROOT / f"experiments/exp{i}_*"

    built: list[Path] = []
    if all or not only or "closed" in (only or []):
        from app.figures.closed import build_closed

        built += build_closed(fig_dir, exps)
        from app.figures.closed import build_heatmap_13

        built += build_heatmap_13(fig_dir, exps)
        typer.echo(f"heatmap_1_3 -> {len(built)} files")
        typer.echo(f"closed -> {len(built)} files")
    if all or (only and "gcd" in only):
        from app.figures.gcd import build_gcd

        try:
            built += build_gcd(fig_dir, exps)
        except Exception as e:
            typer.echo(f"gcd skip: {e}")
    if all or (only and "osr" in only):
        from app.figures.osr import build_osr

        try:
            built += build_osr(fig_dir, exps)
        except Exception as e:
            typer.echo(f"osr skip: {e}")
    if all or (only and "reliability" in only):
        from app.figures.reliability import build_reliability

        try:
            built += build_reliability(fig_dir, exps)
        except Exception as e:
            typer.echo(f"reliability skip: {e}")

    typer.echo(f"figures all={all} -> {len(built)} files in {fig_dir}")
    if sync_frontend:
        from app.figures.sync import sync_figures

        sync_figures(fig_dir)
