#!/usr/bin/env python3
"""Upload NCD results from volume to HF.
Utility script — NCD upload is now handled by the HF Space entrypoint.
This script exists as a fallback for manual sync if needed."""

import logging
import os

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ID = "marcellorusso/orchid-osr-models"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
api = HfApi(token=HF_TOKEN)

# (exp_prefix, model) pairs from the v22 job
MODELS = [
    ("exp8_sphor", "exp8_sphor_dinov2_small"),
    ("exp8_sphor", "exp8_sphor_dinov2_base"),
    ("exp8_sphor", "exp8_sphor_resnet18"),
    ("exp8_sphor", "exp8_sphor_resnet50"),
    ("exp10_mgh_ncd", "exp10_mgh_ncd_dinov2_small"),
    ("exp10_mgh_ncd", "exp10_mgh_ncd_dinov2_base"),
    ("exp10_mgh_ncd", "exp10_mgh_ncd_resnet18"),
    ("exp10_mgh_ncd", "exp10_mgh_ncd_resnet50"),
]

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
EXPERIMENTS_DIR = os.path.abspath(EXPERIMENTS_DIR)

# In OVH the experiments are symlinked from /workspace/data/experiments
OVH_EXP_DIR = "/workspace/backend/experiments"
if os.path.exists(OVH_EXP_DIR):
    EXPERIMENTS_DIR = OVH_EXP_DIR
    logger.info(f"Using OVH experiments dir: {EXPERIMENTS_DIR}")


def main():
    for hf_prefix, exp_name in MODELS:
        exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
        if not os.path.exists(exp_dir):
            logger.warning(f"Skip {exp_name}: directory not found")
            continue

        hf_path = f"{hf_prefix}/{exp_name}"
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
