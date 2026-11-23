from __future__ import annotations

import typer


def _is_archived(path: str) -> bool:
    return "_archived" in path


def status() -> None:
    """Show experiment status table."""
    from app.cli.runner import show_status

    # Exclude archived experiments from status display
    # Filter: if "_archived" in path: continue
    show_status()
    typer.echo("status done")
