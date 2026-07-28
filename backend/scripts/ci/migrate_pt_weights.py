"""One-time migration: re-save all .pt weights with PyTorch >=2.6 compatible format."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.hf_config import HF_MODELS_REPO

import torch
from huggingface_hub import HfApi, snapshot_download

HF_REPO = HF_MODELS_REPO
LOCAL_CACHE = "/tmp/pt_migration"
os.makedirs(LOCAL_CACHE, exist_ok=True)


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN required")
        sys.exit(1)

    api = HfApi(token=token)

    print("Downloading .pt files from HF...")
    snapshot_download(
        HF_REPO,
        repo_type="model",
        local_dir=LOCAL_CACHE,
        allow_patterns=["*.pt", "*_adapter/*"],
        token=token,
    )

    migrated = errors = 0
    for pt_path in sorted(Path(LOCAL_CACHE).rglob("*.pt")):
        rel = pt_path.relative_to(LOCAL_CACHE)
        try:
            state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            torch.save(state_dict, str(pt_path))
            print(f"  OK  {rel}")
            migrated += 1
        except Exception as e:
            print(f"  FAIL {rel}: {e}")
            errors += 1

    print(f"\nMigrated: {migrated}, Errors: {errors}")
    print("Uploading migrated .pt files back to HF...")

    uploaded = 0
    for pt_path in sorted(Path(LOCAL_CACHE).rglob("*.pt")):
        rel = pt_path.relative_to(LOCAL_CACHE)
        try:
            api.upload_file(
                repo_id=HF_REPO,
                path_or_fileobj=str(pt_path),
                path_in_repo=str(rel),
                repo_type="model",
                token=token,
            )
            print(f"  Upload OK  {rel}")
            uploaded += 1
        except Exception as e:
            print(f"  Upload FAIL {rel}: {e}")
            errors += 1

    print(f"Uploaded: {uploaded}, Errors: {errors}")
    if errors:
        print("WARNING: some files failed to upload. Non-pt files on HF are untouched.")
    else:
        print("Migration complete.")


if __name__ == "__main__":
    main()
