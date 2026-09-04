from __future__ import annotations

from typing import Annotated

import typer


def ncd(
    exp: Annotated[int, typer.Option("--exp", help="Exp ID (1-5)")] = 4,
    model: Annotated[str, typer.Option(help="Model variant")] = "resnet18",
) -> None:
    """Run NCD / GCD pipeline (OSR + clustering + figures) on GPU if available."""
    from app.cli.runner import run_experiment

    typer.echo(f"ncd exp={exp} model={model} (GPU auto)")
    # Usa la stessa logica di train ma in fase ncd (estrazione embedding su GPU + clustering SS-KM + OSR)
    run_experiment(exp_id=exp, model_variant=model, phases=["ncd"])
