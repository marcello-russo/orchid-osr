from __future__ import annotations

import typer


def hf(
    action: str = typer.Argument("upload", help="upload|download"),
    model: str = typer.Option("resnet18", help="Model variant"),
) -> None:
    """HuggingFace upload/download."""
    typer.echo(f"hf action={action} model={model}")
