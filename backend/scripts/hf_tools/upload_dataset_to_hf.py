"""
Upload the clean dataset (exp6) to HuggingFace.

Usage:
    python scripts/upload_dataset_to_hf.py

Requires: huggingface_hub, HF token configured (hf auth login)

Repo: marcellorusso/orchid-ncd-dataset
"""
import os
import shutil
from huggingface_hub import HfApi, create_repo

REPO_ID = "marcellorusso/orchid-ncd-dataset"
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


def main():
    api = HfApi()

    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"Repo {REPO_ID} ready")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Create a real staging directory (not symlinks) in a temp location
    import tempfile
    staging = tempfile.mkdtemp(prefix="hf_upload_")
    try:
        os.makedirs(os.path.join(staging, "train_clean"), exist_ok=True)
        os.makedirs(os.path.join(staging, "test_clean"), exist_ok=True)

        # Copy train_clean
        train_src = os.path.join(DATASET_DIR, "train_clean")
        print("Copying train_clean to staging...")
        for cls in os.listdir(train_src):
            src_cls = os.path.join(train_src, cls)
            if not os.path.isdir(src_cls):
                continue
            dst_cls = os.path.join(staging, "train_clean", cls)
            print(f"  {cls}...")
            shutil.copytree(src_cls, dst_cls)

        # Copy test_clean
        test_src = os.path.join(DATASET_DIR, "test_clean")
        print("Copying test_clean to staging...")
        for cls in os.listdir(test_src):
            src_cls = os.path.join(test_src, cls)
            if not os.path.isdir(src_cls):
                continue
            dst_cls = os.path.join(staging, "test_clean", cls)
            print(f"  {cls}...")
            shutil.copytree(src_cls, dst_cls)

        # Copy metadata files
        for fname, src in [
            ("split_manifest.json", os.path.join(DATASET_DIR, "split_manifest.json")),
            ("create_clean_split.py", os.path.join(os.path.dirname(__file__), "create_clean_split.py")),
            ("README.md", os.path.join(os.path.dirname(__file__), "hf_dataset_readme.md")),
        ]:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(staging, fname))

        total_files = sum(len(f) for _, _, f in os.walk(staging))
        print(f"\nUploading {total_files} files from {staging}")
        print(f"Contents: {os.listdir(staging)}")

        api.upload_large_folder(
            repo_id=REPO_ID,
            folder_path=staging,
            repo_type="dataset",
        )
        print("Upload complete")

    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\nDone! Dataset at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
