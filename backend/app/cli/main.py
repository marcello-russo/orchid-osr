from __future__ import annotations

import typer

from app.cli.commands.figures import figures
from app.cli.commands.hf import hf_app
from app.cli.commands.ncd import ncd, ncd_comparison
from app.cli.commands.status import status
from app.cli.commands.train import train

app = typer.Typer(rich_markup_mode="rich", add_completion=True, help="OrchID — ultra-fine-grained OSR")


app.command()(train)
app.command()(ncd)
app.command(name="ncd-comparison")(ncd_comparison)
app.command()(figures)
app.add_typer(hf_app, name="hf")
app.command()(status)


@app.command("embeddings")
def embeddings(
    exp: int = typer.Option(..., "--exp", help="Exp 1-5", min=1, max=5),
    model: str = typer.Option("resnet18", help="Model variant"),
) -> None:
    """Alias to train --embeddings (extract embeddings for latent viewer)."""
    train(exp=exp, model=model, embeddings=True)


if __name__ == "__main__":
    app()
