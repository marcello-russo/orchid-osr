from __future__ import annotations

from typing import Annotated

import typer

hf_app = typer.Typer(help="HuggingFace upload/download/sync.", no_args_is_help=True)


@hf_app.command("upload-models")
def upload_models(
    exp: Annotated[str, typer.Option("--exp", help="all or experiment filter")] = "all",
    cleanup: Annotated[bool, typer.Option("--cleanup", help="Delete superseded remote files")] = False,
) -> None:
    """Upload best-fold weights + metadata to the HF models repo."""
    from app.hf.models import run_upload

    n = run_upload(exp_filter=exp, do_cleanup=cleanup)
    typer.echo(f"uploaded {n} models")


@hf_app.command("upload-ncd")
def upload_ncd_cmd() -> None:
    """Upload ncd_results.json + ncd_figures to the HF models repo."""
    from huggingface_hub import HfApi

    from app.config import settings
    from app.hf.ncd import upload_ncd

    api = HfApi(token=settings.hf_token or None)
    n = upload_ncd(api)
    typer.echo(f"uploaded {n} NCD files")


@hf_app.command("sync")
def sync_cmd() -> None:
    """Download essential experiment JSONs + embeddings from HF."""
    from app.hf.sync import sync_experiments

    out = sync_experiments()
    typer.echo(f"synced={out['synced']} skipped={out['skipped']} errors={out['errors']}")


@hf_app.command("download")
def download_cmd(
    sample_ratio: Annotated[float, typer.Option("--sample-ratio", help="Keep fraction per class dir")] = 1.0,
) -> None:
    """Download + reshape the cropped dataset from HF."""
    from app.hf.sync import download_dataset

    ok = download_dataset(sample_ratio=sample_ratio)
    typer.echo(f"dataset download {'ok' if ok else 'failed'}")
