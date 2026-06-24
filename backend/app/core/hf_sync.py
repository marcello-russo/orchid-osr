"""Shared HF download utilities for experiment sync."""
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def download_file(api: HfApi, repo: str, src_path: str, dst_path: str | Path, token: str):
    """Download a single file from HF with atomic .tmp+os.replace."""
    from huggingface_hub import hf_hub_download

    dl = hf_hub_download(repo, src_path, token=token)
    tmp = str(dst_path) + ".tmp"
    shutil.copy2(dl, tmp)
    os.replace(tmp, str(dst_path))


def list_repo_files(api: HfApi, repo: str) -> list[str]:
    """List all files in HF repo, return filtered paths."""
    try:
        items = list(api.list_repo_tree(repo, recursive=True))
    except Exception as e:
        print(f"Failed to list repo: {e}")
        return []
    return [i.path for i in items if "." in i.path.split("/")[-1]]
