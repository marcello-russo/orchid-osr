import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.hf_config import HF_DATASET_REPO

from huggingface_hub import snapshot_download

DATASET_REPO = HF_DATASET_REPO

SAMPLE_RATIO = float(os.environ.get("DATASET_SAMPLE_RATIO", "1.0"))


def _load_token() -> str | None:
    secret_path = "/run/secrets/HF_TOKEN"
    if os.path.exists(secret_path):
        with open(secret_path) as f:
            return f.read().strip()
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        return os.environ.get("HF_TOKEN")
    return None


def _sample_directory(path: str, ratio: float):
    """Keep only `ratio` of files in each class subdirectory."""
    if ratio >= 1.0:
        return
    base = Path(path)
    if not base.exists():
        return
    for class_dir in base.iterdir():
        if not class_dir.is_dir():
            continue
        files = [f for f in class_dir.iterdir() if f.is_file()]
        random.shuffle(files)
        keep = max(1, int(len(files) * ratio))
        for f in files[keep:]:
            f.unlink()
        print(f"  {class_dir.name}: {len(files)} → {keep} file (ratio={ratio})")


def main():
    token = _load_token()
    if not token:
        print("Warning: HF_TOKEN not found. Downloads might hit rate limits or fail if repos are private.")

    # --- Dataset ---
    print("Downloading dataset from HuggingFace...")
    try:
        snapshot_download(
            DATASET_REPO,
            repo_type="dataset",
            local_dir="dataset",
            allow_patterns=[
                "cropped/*/train/**",
                "cropped/*/test/**",
                "cropped/*/test_clean/**",
            ],
            token=token,
        )

        for size in ["256x512", "224x224"]:
            for hf_split in ["train", "test", "test_clean"]:
                src = os.path.join("dataset", "cropped", size, hf_split)
                dst = os.path.join("dataset", f"cropped_{size}", hf_split.replace("_clean", ""))
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    for item in os.listdir(src):
                        s = os.path.join(src, item)
                        t = os.path.join(dst, item)
                        if os.path.isdir(s):
                            shutil.copytree(s, t, dirs_exist_ok=True)
                    shutil.rmtree(src)
                    print(f"  cropped/{size}/{hf_split} -> cropped_{size}/{hf_split.replace('_clean', '')}")

        # Sample dataset if ratio < 1.0
        if SAMPLE_RATIO < 1.0:
            print(f"Dataset sampling attivo: ratio={SAMPLE_RATIO}")
            for size in ["256x512", "224x224"]:
                for split in ["train", "test"]:
                    p = os.path.join("dataset", f"cropped_{size}", split)
                    if os.path.isdir(p):
                        print(f"  Campionamento {p}...")
                        _sample_directory(p, SAMPLE_RATIO)

        print("Dataset downloaded successfully.")
    except Exception as e:
        print(f"Dataset download failed: {e}")


if __name__ == "__main__":
    main()
