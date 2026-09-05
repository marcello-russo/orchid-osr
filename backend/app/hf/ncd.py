"""NCD results upload — importable logic (CLI: `orchid hf upload-ncd`)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

from app.config import settings

logger = logging.getLogger(__name__)

REPO_ID = settings.hf_models_repo
DEFAULT_EXP_BASE = str(Path(__file__).resolve().parent.parent.parent / "experiments")

# (local_dir_prefix, hf_root) pairs — hf_root = remote HF root folder.
MODELS = [
    ("exp3_sphor", "exp8_sphor"),
    ("exp4_sphor_ncd", "exp9_ncd"),
    ("exp5_sphor_ncd", "exp5_sphor_ncd"),
    ("exp10_mgh_ncd", "exp10_mgh_ncd"),
]


def _resolve_exp_base(exp_base: str | None) -> str:
    if exp_base:
        return exp_base
    ovh = "/workspace/backend/experiments"
    if os.path.exists(ovh):
        logger.info(f"Using OVH experiments dir: {ovh}")
        return ovh
    return DEFAULT_EXP_BASE


def upload_ncd(api: HfApi, exp_base: str | None = None, repo: str = REPO_ID) -> int:
    """Upload ncd_results.json + ncd_figures per experiment dir. Returns file count."""
    experiments_dir = _resolve_exp_base(exp_base)
    n = 0
    for local_prefix, hf_root in MODELS:
        matches = [
            d
            for d in os.listdir(experiments_dir)
            if d.startswith(local_prefix + "_") and os.path.isdir(os.path.join(experiments_dir, d))
        ]
        if not matches:
            logger.warning(f"No local dirs found for prefix {local_prefix}")
            continue

        for exp_name in sorted(matches):
            exp_dir = os.path.join(experiments_dir, exp_name)
            if not os.path.exists(exp_dir):
                continue
            remote_dir_name = exp_name.replace(local_prefix, hf_root, 1)
            hf_path = f"{hf_root}/{remote_dir_name}"
            logger.info(f"Uploading NCD results to {hf_path}")

            ncd_json = os.path.join(exp_dir, "ncd_results.json")
            if os.path.exists(ncd_json):
                api.upload_file(
                    repo_id=repo,
                    path_in_repo=f"{hf_path}/ncd_results.json",
                    path_or_fileobj=ncd_json,
                )
                logger.info("  -> ncd_results.json")
                n += 1

            ncd_figures = os.path.join(exp_dir, "ncd_figures")
            if os.path.isdir(ncd_figures):
                for fname in os.listdir(ncd_figures):
                    fpath = os.path.join(ncd_figures, fname)
                    if not os.path.isfile(fpath):
                        continue
                    api.upload_file(
                        repo_id=repo,
                        path_in_repo=f"{hf_path}/ncd_figures/{fname}",
                        path_or_fileobj=fpath,
                    )
                    n += 1
                logger.info(f"  -> ncd_figures/ ({len(os.listdir(ncd_figures))} files)")

    comp_path = os.path.join(experiments_dir, "ncd_comparison.json")
    if os.path.exists(comp_path):
        api.upload_file(
            repo_id=repo,
            path_in_repo="ncd_comparison.json",
            path_or_fileobj=comp_path,
        )
        logger.info("Uploaded ncd_comparison.json")
        n += 1

    logger.info("Done!")
    return n


def main() -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN", "") or settings.hf_token or None)
    upload_ncd(api)


if __name__ == "__main__":
    main()
