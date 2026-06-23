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
    DST = "/data/experiments"

    os.makedirs(DST, exist_ok=True)

    try:
        items = list(api.list_repo_tree(REPO, recursive=True))
    except Exception as e:
        print(f"Failed to list repo: {e}")
        return

    paths = [i.path for i in items if "." in i.path.split("/")[-1]]
    count = 0

    for src in paths:
        parts = src.split("/")
        if len(parts) < 2 or parts[0] in ("old",):
            continue
        flat = "/".join(parts[1:])
        dst = os.path.join(DST, flat)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            dl = hf_hub_download(REPO, src, token=token)
            shutil.copy2(dl, dst)
            count += 1
        except Exception:
            pass

    try:
        p = hf_hub_download(REPO, "ncd_comparison.json", token=token)
        shutil.copy2(p, os.path.join(DST, "ncd_comparison.json"))
    except Exception:
        pass

    print(f"Downloaded {count} files")


if __name__ == "__main__":
    main()
