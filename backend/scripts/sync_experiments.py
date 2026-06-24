"""Download experiment results from HuggingFace models repo at container startup."""
import os
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("No HF_TOKEN, skipping experiment sync")
        return

    api = HfApi(token=token)
    REPO = "marcellorusso/orchid-ncd-models"
    DST = os.environ.get("EXPERIMENTS_DIR", "/data/experiments")

    os.makedirs(DST, exist_ok=True)

    try:
        items = list(api.list_repo_tree(REPO, recursive=True))
    except Exception as e:
        print(f"Failed to list repo: {e}")
        return

    paths = [i.path for i in items if "." in i.path.split("/")[-1]]
    count = 0
    skipped = 0

    for src in paths:
        parts = src.split("/")
        if len(parts) < 2 or parts[0] in ("old",):
            continue
        flat = "/".join(parts[1:])
        dst = os.path.join(DST, flat)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # Incremental: skip if exists (except JSON files which may be updated)
        if not src.endswith(".json") and os.path.exists(dst):
            skipped += 1
            continue

        try:
            dl = hf_hub_download(REPO, src, token=token)
            tmp = str(dst) + ".tmp"
            shutil.copy2(dl, tmp)
            os.replace(tmp, dst)
            count += 1
        except Exception as e:
            print(f"  WARN: failed to download {src}: {e}")

    # ncd_comparison.json at repo root (not under any exp dir)
    try:
        ncd_dst = os.path.join(DST, "ncd_comparison.json")
        p = hf_hub_download(REPO, "ncd_comparison.json", token=token)
        tmp = ncd_dst + ".tmp"
        shutil.copy2(p, tmp)
        os.replace(tmp, ncd_dst)
        count += 1
    except Exception as e:
        print(f"  WARN: failed to download ncd_comparison.json: {e}")

    print(f"Downloaded {count} files (skipped {skipped} existing)")


if __name__ == "__main__":
    main()
