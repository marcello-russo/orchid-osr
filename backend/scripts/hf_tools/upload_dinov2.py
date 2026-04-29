import os
from huggingface_hub import HfApi

def upload_model():
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try reading from common secret path if in container
        secret_path = "/run/secrets/HF_TOKEN"
        if os.path.exists(secret_path):
            with open(secret_path, "r") as f:
                token = f.read().strip()
    
    if not token:
        print("Error: HF_TOKEN not found.")
        return

    repo_id = "marcellorusso/orchid-ncd-models"
    base_path = "backend/experiments/exp6_clean_split_dinov2_base"
    remote_base = "serie1_ce/exp6_clean_split_dinov2_base"

    print(f"Uploading files to {repo_id}...")

    # Upload the .pt file
    api.upload_file(
        path_or_fileobj=os.path.join(base_path, "best_fold_1.pt"),
        path_in_repo=f"{remote_base}/best_fold_1.pt",
        repo_id=repo_id,
        token=token
    )
    print("Uploaded best_fold_1.pt")

    # Upload the results.json for the downloader to find it
    api.upload_file(
        path_or_fileobj=os.path.join(base_path, "results.json"),
        path_in_repo=f"{remote_base}/results.json",
        repo_id=repo_id,
        token=token
    )
    print("Uploaded results.json")

    # Upload the adapter folder
    adapter_path = os.path.join(base_path, "best_fold_1_adapter")
    for root, _, files in os.walk(adapter_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, base_path)
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=f"{remote_base}/{relative_path}",
                repo_id=repo_id,
                token=token
            )
            print(f"Uploaded {relative_path}")

if __name__ == "__main__":
    upload_model()
