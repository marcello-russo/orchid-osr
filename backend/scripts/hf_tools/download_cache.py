import os
import shutil
from huggingface_hub import snapshot_download

DATASET_REPO = "marcellorusso/orchid-ncd-dataset"
MODELS_REPO = "marcellorusso/orchid-ncd-models"

EXP_PATHS = ["exp6_ce", "exp7_supcon", "exp8_sphor", "exp9_ncd", "exp10_mgh_ncd"]


def _download_and_flatten_models(local_base, token):
    """Download all experiments from HF and flatten into experiments/."""
    snapshot_download(
        MODELS_REPO,
        repo_type="model",
        local_dir=local_base,
        allow_patterns=[
            "exp6_ce/**/*.pt",
            "exp6_ce/**/*.json",
            "exp6_ce/**/*.safetensors",
            "exp6_ce/**/embeddings/*",
            "exp7_supcon/**/*.pt",
            "exp7_supcon/**/*.json",
            "exp7_supcon/**/*.safetensors",
            "exp7_supcon/**/embeddings/*",
            "exp8_sphor/**/*.pt",
            "exp8_sphor/**/*.json",
            "exp8_sphor/**/*.safetensors",
            "exp8_sphor/**/embeddings/*",
            "exp9_ncd/**/*.pt",
            "exp9_ncd/**/*.json",
            "exp9_ncd/**/*.safetensors",
            "exp9_ncd/**/embeddings/*",
        ],
        token=token,
    )

    for exp in EXP_PATHS:
        src = os.path.join(local_base, exp)
        if not os.path.isdir(src):
            print(f"  {exp}/ not found on HF, skipping")
            continue
        for d in os.listdir(src):
            s = os.path.join(src, d)
            t = os.path.join(local_base, d)
            if os.path.isdir(s):
                shutil.copytree(s, t, dirs_exist_ok=True)
                print(f"  Merged {exp}/{d}")
        shutil.rmtree(src)

    open(os.path.join(local_base, ".models_synced"), "w").close()
    print("Models synced successfully.")


def _load_token() -> str | None:
    """Load HF_TOKEN from Docker secret, env, or .env file."""
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
            allow_patterns=[
                "cropped/*/train/*",
                "cropped/*/test/*",
                "cropped/*/test_clean/*",
            ],
            token=token,
        )

        for size in ["256x512", "224x224"]:
            # Merge each HF split into the local directory (union of train + test + test_clean species)
            for hf_split in ["train", "test", "test_clean"]:
                src = os.path.join("dataset", "cropped", size, hf_split)
                dst = os.path.join("dataset", f"cropped_{size}", "test_clean" if hf_split in ("test", "test_clean") else "train_clean")
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    for item in os.listdir(src):
                        s = os.path.join(src, item)
                        t = os.path.join(dst, item)
                        if os.path.isdir(s):
                            shutil.copytree(s, t, dirs_exist_ok=True)
                    shutil.rmtree(src)
                    print(f"  cropped/{size}/{hf_split} merged into cropped_{size}/{'test_clean' if hf_split in ('test', 'test_clean') else 'train_clean'}")

        print("Dataset downloaded successfully.")
    except Exception as e:
        print(f"Dataset download failed: {e}")

    # --- Models ---
    print("\nDownloading model weights from HuggingFace...")
    try:
        _download_and_flatten_models("experiments", token)
    except Exception as e:
        print(f"Model download failed: {e}")


if __name__ == "__main__":
    main()
