"""Shared HF download utilities for experiment sync."""

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def download_file(api: HfApi, repo: str, src_path: str, dst_path: str | Path, token: str):
    """Download a single file from HF with atomic .tmp+os.replace."""

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


def parse_exp_dir_name(dir_name: str, model_set: set[str] | None = None) -> tuple[int | None, str | None]:
    """Parse experiment dir name like 'exp10_mgh_ncd_convnext_small' → (10, 'convnext_small').

    Returns (exp_num, model_variant) or (None, None) if parsing fails.
    """
    if model_set is None:
        model_set = {
            "resnet18",
            "resnet50",
            "convnext_tiny",
            "convnext_small",
            "dinov2_small",
            "dinov2_base",
        }
    parts = dir_name.split("_")
    for i, p in enumerate(parts):
        if p.startswith("exp") and p[3:].isdigit():
            exp_num = int(p[3:])
            for j in range(len(parts) - 1, i, -1):
                candidate = "_".join(parts[j:])
                if candidate in model_set:
                    return exp_num, candidate
            return exp_num, None
    return None, None
