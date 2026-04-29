import os
import shutil
from huggingface_hub import snapshot_download

def main():
    # Read token from Docker BuildKit secret
    token = None
    secret_path = "/run/secrets/HF_TOKEN"
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            token = f.read().strip()
    else:
        # Fallback to env var if running locally without secrets
        token = os.environ.get("HF_TOKEN")

    if not token:
        print("Warning: HF_TOKEN not found in secrets or environment. Downloads might hit rate limits or fail if repositories are private.")

    print("Downloading clean dataset from HuggingFace...")
    try:
        snapshot_download(
            'marcellorusso/orchid-ncd-dataset',
            repo_type='dataset',
            local_dir='dataset',
            allow_patterns=['train_clean/*', 'test_clean/*', 'split_manifest.json'],
            token=token
        )
        print("Dataset downloaded successfully.")
    except Exception as e:
        print(f"Dataset download failed: {e}")

    print("\nDownloading model weights from HuggingFace...")
    try:
        snapshot_download(
            'marcellorusso/orchid-ncd-models',
            repo_type='model',
            local_dir='experiments',
            allow_patterns=['serie1_ce/**/*.pt', 'serie1_ce/**/*.json', 'serie1_ce/**/*.safetensors'],
            token=token
        )
        
        # Flatten: move serie1_ce/exp6_*/ to experiments/exp6_*/
        src = 'experiments/serie1_ce'
        if os.path.isdir(src):
            for d in os.listdir(src):
                s = os.path.join(src, d)
                t = os.path.join('experiments', d)
                if os.path.isdir(s):
                    shutil.copytree(s, t, dirs_exist_ok=True)
                    print(f"Merged {d}")
            shutil.rmtree(src)
            
        # Create marker file to signal successful sync
        open('experiments/.models_synced', 'w').close()
        print("Models synced successfully.")
    except Exception as e:
        print(f"Model download failed: {e}")

if __name__ == "__main__":
    main()
