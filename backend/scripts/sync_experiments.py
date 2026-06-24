"""Download experiment results from HuggingFace models repo at container startup."""
import os

from huggingface_hub import HfApi

from app.core.hf_sync import download_file, list_repo_files


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("No HF_TOKEN, skipping experiment sync")
        return

    api = HfApi(token=token)
    REPO = "marcellorusso/orchid-ncd-models"
    DST = os.environ.get("EXPERIMENTS_DIR", "/data/experiments")
    os.makedirs(DST, exist_ok=True)

    paths = list_repo_files(api, REPO)
    if not paths:
        return

    count = 0
    skipped = 0

    for src in paths:
        parts = src.split("/")
        if parts[0] in ("old",):
            continue
        if len(parts) < 2:
            flat = src  # root-level file (ncd_comparison.json)
        else:
            flat = "/".join(parts[1:])
        dst = os.path.join(DST, flat)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if not src.endswith(".json") and os.path.exists(dst):
            skipped += 1
            continue

        try:
            download_file(api, REPO, src, dst, token)
            count += 1
        except Exception as e:
            print(f"  WARN: failed to download {src}: {e}")

    print(f"Downloaded {count} files (skipped {skipped} existing)")


if __name__ == "__main__":
    main()
