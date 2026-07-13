import os
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_REPO = "marcellorusso/orchid-ncd-dataset"
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
            resume=True,
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

        print("Dataset downloaded successfully.")
    except Exception as e:
        print(f"Dataset download failed: {e}")

if __name__ == "__main__":
    main()
