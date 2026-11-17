from __future__ import annotations

from typing import Annotated

import typer


def ncd(
    exp: Annotated[int, typer.Option("--exp", help="Exp ID (1-5)")] = 4,
    model: Annotated[str, typer.Option(help="Model variant")] = "resnet18",
    stats: Annotated[bool, typer.Option("--stats", help="Only build osr_stats.npz calibration")] = False,
    delong: Annotated[bool, typer.Option("--delong", help="Only print DeLong report (needs osr_scores.npz)")] = False,
) -> None:
    """Run NCD / GCD pipeline (OSR + clustering + figures) on GPU if available."""
    if stats:
        from app.core.ncd.stats import run_stats

        results = run_stats()
        typer.echo(f"stats built for {len(results)} models")
        return
    if delong:
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent.parent / "scripts" / "compute_delong.py"
        # Lista di argomenti, nessuna shell e nessun input utente: il comando è
        # questo interprete su uno script del repo.
        raise SystemExit(subprocess.call([sys.executable, str(script)]))  # noqa: S603
    from app.cli.runner import run_experiment

    typer.echo(f"ncd exp={exp} model={model} (GPU auto)")
    # Usa la stessa logica di train ma in fase ncd (estrazione embedding su GPU + clustering SS-KM + OSR)
    run_experiment(exp_id=exp, model_variant=model, phases=["ncd"])


def ncd_comparison() -> None:
    """Aggregate NCD results into ncd_comparison.json."""
    from app.cli.definitions import PROJECT_ROOT
    from app.core.ncd.comparison import build_comparison

    entries = build_comparison(str(PROJECT_ROOT / "experiments"))
    typer.echo(f"ncd_comparison: {len(entries)} entries")
