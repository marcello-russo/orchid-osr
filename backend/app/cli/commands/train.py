from __future__ import annotations

from typing import Annotated

import typer


def train(
    exp: Annotated[int, typer.Option("--exp", help="Exp 1-5", min=1, max=5)],
    model: Annotated[
        str, typer.Option(help="resnet18|resnet50|convnext_tiny|convnext_small|dinov2_small|dinov2_base")
    ] = "resnet18",
    sphor: Annotated[bool, typer.Option(help="Abilita SpHOR (solo Exp 3-5)")] = False,
    warmup_epochs: Annotated[int, typer.Option(help="Epoche di solo SupCon prima di attivare i prototipi SpHOR")] = 20,
    vmf_kappa: Annotated[float, typer.Option(help="Concentrazione kappa della loss vMF (hard assignment)")] = 10.0,
    ramp_epochs: Annotated[int, typer.Option(help="Epoche su cui il peso della vMF cresce 0 -> 1")] = 20,
    repulse_weight: Annotated[float, typer.Option(help="Peso della repulsione ortogonale tra prototipi")] = 0.01,
    novel: Annotated[str | None, typer.Option(help="Override novel: garganica,majellensis")] = None,
    profile: Annotated[str, typer.Option(help="cpu|gpu")] = "gpu",
    embeddings: Annotated[bool, typer.Option(help="Extract embeddings (alias)")] = False,
) -> None:
    """Train / cross-validate an experiment (k-fold)."""
    from app.cli.definitions import parse_novel_arg, resolve_experiment
    from app.cli.runner import run_experiment

    override = parse_novel_arg(novel) if novel else None
    cfg = resolve_experiment(exp, override)
    typer.echo(f"Exp {exp} -> {cfg['name']}")
    if embeddings:
        typer.echo("embeddings mode")
    run_experiment(
        exp_id=exp,
        model_variant=model,
        novel=novel,
        use_prototypes=sphor,
        warmup_epochs=warmup_epochs,
        vmf_kappa=vmf_kappa,
        ramp_epochs=ramp_epochs,
        repulse_weight=repulse_weight,
    )
