#!/usr/bin/env python3
"""Upload experiment JSON results (exp8/exp9 new runs) to HF models repo.

Uploads results.json, config.json, fold_*.json, ncd_results.json for the
freshly completed exp8/exp9 runs. Keeps exp6/7/10 untouched.
"""
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
from scripts.hf_config import HF_MODELS_REPO

from dotenv import load_dotenv
from huggingface_hub import HfApi

_env = PROJECT_DIR / ".env"
if _env.exists():
    load_dotenv(_env)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("HF_TOKEN non trovato. Abort.")
    sys.exit(1)

EXP_BASE = BACKEND_DIR / "experiments"
MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]

api = HfApi(token=HF_TOKEN)
uploaded = 0
skipped = 0
failed = 0

for exp_prefix in ["exp8_sphor", "exp9_ncd"]:
    for model in MODELS:
        exp_name = f"{exp_prefix}_{model}"
        local_dir = EXP_BASE / exp_name
        if not local_dir.is_dir():
            print(f"SKIP {exp_name}: dir non esiste")
            skipped += 1
            continue

        results_path = local_dir / "results.json"
        if not results_path.exists():
            print(f"SKIP {exp_name}: results.json mancante")
            skipped += 1
            continue

        remote_base = f"{exp_prefix}/{exp_name}"
        # Upload results.json + config.json + fold_*.json
        for pattern in ["results.json", "config.json"]:
            fp = local_dir / pattern
            if fp.exists():
                api.upload_file(repo_id=HF_MODELS_REPO, path_in_repo=f"{remote_base}/{pattern}", path_or_fileobj=str(fp))
                uploaded += 1
                print(f"  UP {remote_base}/{pattern}")

        for fp in sorted(local_dir.glob("fold_*.json")):
            api.upload_file(repo_id=HF_MODELS_REPO, path_in_repo=f"{remote_base}/{fp.name}", path_or_fileobj=str(fp))
            uploaded += 1
            print(f"  UP {remote_base}/{fp.name}")

        # Upload ncd_results.json for exp9
        ncd = local_dir / "ncd_results.json"
        if ncd.exists():
            api.upload_file(repo_id=HF_MODELS_REPO, path_in_repo=f"{remote_base}/ncd_results.json", path_or_fileobj=str(ncd))
            uploaded += 1
            print(f"  UP {remote_base}/ncd_results.json")

print(f"\nDone: {uploaded} file caricati, {skipped} saltati, {failed} falliti")
