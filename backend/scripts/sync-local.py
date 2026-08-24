#!/usr/bin/env python3
"""Sync local experiments from HuggingFace models repo.
Downloads missing config.json, results.json, fold_*.json, best_fold_N.pt
for experiment directories that lack these metadata files.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from huggingface_hub import HfApi

from app.core.hf_sync import download_file, list_repo_files
from scripts.hf_config import HF_MODELS_REPO


def main():
    backend_dir = Path(__file__).resolve().parent.parent
    exp_dir = Path(os.environ.get("EXP_DIR", str(backend_dir / "experiments")))

    # Load .env from project root
    dotenv_path = backend_dir.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set and not found in .env")
        sys.exit(1)

    print(f"Syncing experiments from {HF_MODELS_REPO}...")
    print(f"Experiment dir: {exp_dir}")

    if not exp_dir.is_dir():
        print(f"Experiment directory not found: {exp_dir}")
        sys.exit(1)
    if not exp_dir.parent.is_dir():
        print(f"Backend directory not found: {exp_dir.parent}, check EXP_DIR")
        sys.exit(1)

    api = HfApi(token=token)
    REPO = HF_MODELS_REPO

    print("Listing HF repo contents...")
    hf_paths = list_repo_files(api, REPO)
    if not hf_paths:
        return
    print(f"Found {len(hf_paths)} files on HF")

    # Build set of exp dirs on HF
    hf_exps: dict[str, list[str]] = {}
    for p in hf_paths:
        parts = p.split("/")
        if len(parts) >= 2 and parts[0] not in ("old",):
            exp_name = parts[1]
            hf_exps.setdefault(exp_name, []).append(p)

    print(f"Found {len(hf_exps)} experiment dirs on HF")

    # Scan local experiment dirs
    local_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("exp")])
    synced = 0
    skipped = 0

    for d in local_dirs:
        name = d.name
        has_config = (d / "config.json").exists()
        has_results = (d / "results.json").exists()
        has_best_pt = any(
            f.name.startswith("best_fold_") and (f.name.endswith(".pt") or f.name.endswith(".safetensors"))
            for f in d.iterdir()
        )

        needs_config = not has_config
        needs_results = not has_results
        needs_best = not has_best_pt

        if not needs_config and not needs_results and not needs_best:
            continue

        if name not in hf_exps:
            print(f"  SKIP {name}: not found on HF")
            skipped += 1
            continue

        print(f"  {name}: missing config={needs_config} results={needs_results} best_pt={needs_best}")

        for hf_path in hf_exps[name]:
            filename = hf_path.split("/")[-1]

            if filename.startswith("REPORT_"):
                continue
            if filename == "config.json" and not needs_config:
                continue
            if filename == "results.json" and not needs_results:
                continue
            if filename.startswith("best_fold_") and filename.endswith(".pt") and not needs_best:
                continue
            if filename.startswith("fold_") and filename.endswith(".json") and not needs_results:
                continue

            path_parts = hf_path.split("/")
            if any(
                p.startswith("embeddings") or p == "logs" or (p.startswith("best_fold_") and "_adapter" in p)
                for p in path_parts
            ):
                continue

            dst = d / filename
            if dst.exists():
                continue
            try:
                download_file(api, REPO, hf_path, dst, token)
                print(f"    Downloaded {filename}")
                synced += 1
            except Exception as e:
                print(f"    FAILED {filename}: {e}")

    # ncd_comparison.json at repo root
    try:
        comp_dst = exp_dir / "ncd_comparison.json"
        if not comp_dst.exists():
            download_file(api, REPO, "ncd_comparison.json", comp_dst, token)
            synced += 1
            print("    Downloaded ncd_comparison.json")
        else:
            skipped += 1
    except Exception as e:
        print(f"    FAILED ncd_comparison.json: {e}")

    print(f"\nSynced {synced} files, skipped {skipped} experiments")


if __name__ == "__main__":
    main()
