#!/usr/bin/env python3
"""Upload NCD results from volume to HF.
Utility script — NCD upload is now handled by the HF Space entrypoint.
This script exists as a fallback for manual sync if needed."""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from huggingface_hub import HfApi

from scripts.hf_config import HF_MODELS_REPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ID = HF_MODELS_REPO
HF_TOKEN = os.environ.get("HF_TOKEN", "")
api = HfApi(token=HF_TOKEN)

# (local_dir_prefix, hf_root) pairs — hf_root = remote HF root folder, local_dir_prefix = local dir prefix.
MODELS = [
    ("exp3_sphor", "exp8_sphor"),
    ("exp4_sphor_ncd", "exp9_ncd"),
    ("exp10_mgh_ncd", "exp10_mgh_ncd"),
]

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
EXPERIMENTS_DIR = os.path.abspath(EXPERIMENTS_DIR)

# In OVH the experiments are symlinked from /workspace/data/experiments
OVH_EXP_DIR = "/workspace/backend/experiments"
if os.path.exists(OVH_EXP_DIR):
    EXPERIMENTS_DIR = OVH_EXP_DIR
    logger.info(f"Using OVH experiments dir: {EXPERIMENTS_DIR}")


def main():
    for local_prefix, hf_root in MODELS:
        # find all local experiment dirs starting with local_prefix
        matches = [d for d in os.listdir(EXPERIMENTS_DIR)
                   if d.startswith(local_prefix + "_") and os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
        if not matches:
            logger.warning(f"No local dirs found for prefix {local_prefix}")
            continue

        for exp_name in sorted(matches):
            exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
            if not os.path.exists(exp_dir):
                continue
            # remote dir name: replace local prefix with the HF prefix (e.g. exp3_sphor_* -> exp8_sphor_*)
            remote_dir_name = exp_name.replace(local_prefix, hf_root, 1)
            hf_path = f"{hf_root}/{remote_dir_name}"
            logger.info(f"Uploading NCD results to {hf_path}")

            # Upload ncd_results.json
            ncd_json = os.path.join(exp_dir, "ncd_results.json")
            if os.path.exists(ncd_json):
                api.upload_file(
                    repo_id=REPO_ID,
                    path_in_repo=f"{hf_path}/ncd_results.json",
                    path_or_fileobj=ncd_json,
                )
                logger.info("  -> ncd_results.json")

            # Upload ncd_figures/
            ncd_figures = os.path.join(exp_dir, "ncd_figures")
            if os.path.isdir(ncd_figures):
                for fname in os.listdir(ncd_figures):
                    fpath = os.path.join(ncd_figures, fname)
                    if not os.path.isfile(fpath):
                        continue
                    api.upload_file(
                        repo_id=REPO_ID,
                        path_in_repo=f"{hf_path}/ncd_figures/{fname}",
                        path_or_fileobj=fpath,
                    )
                logger.info(f"  -> ncd_figures/ ({len(os.listdir(ncd_figures))} files)")

    # Upload ncd_comparison.json
    comp_path = os.path.join(EXPERIMENTS_DIR, "ncd_comparison.json")
    if os.path.exists(comp_path):
        api.upload_file(
            repo_id=REPO_ID,
            path_in_repo="ncd_comparison.json",
            path_or_fileobj=comp_path,
        )
        logger.info("Uploaded ncd_comparison.json")

    logger.info("Done!")


if __name__ == "__main__":
    main()
