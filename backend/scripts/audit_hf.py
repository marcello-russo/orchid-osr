"""Audit HF repo file ages and flag outdated files."""

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import HfApi

from scripts.hf_config import HF_MODELS_REPO

api = HfApi()
REPO = HF_MODELS_REPO
CUTOFF_DAYS = 15

tree = api.list_repo_tree(REPO, recursive=True, expand=True)
now = datetime.datetime.now()
old = []

for item in tree:
    if not hasattr(item, "is_file") or not item.is_file:
        continue
    age_days = (now - item.last_commit_date).days if item.last_commit_date else 999
    if age_days > CUTOFF_DAYS:
        old.append((age_days, item.path, item.size))

print(f"Files older than {CUTOFF_DAYS} days: {len(old)}")
print()

# Group by type
for age, path, size in sorted(old, reverse=True):
    size_mb = size / (1024 * 1024)
    tag = ""
    if "embeddings" in path:
        tag = " [EMBEDDING]"
    elif path.endswith(".npy"):
        tag = " [NPY]"
    elif path.endswith(".pt"):
        tag = " [PT]"
    elif path.endswith("results.json"):
        tag = " [RESULTS]"
    elif "ncd_" in path:
        tag = " [NCD]"
    elif "fold_" in path:
        tag = " [FOLD]"
    elif "adapter" in path:
        tag = " [ADAPTER]"
    elif "config.json" in path:
        tag = " [CONFIG]"
    print(f"  {age:3d}d {size_mb:7.2f}MB  {path}{tag}")

print(f"\nTotal outdated: {len(old)} files")
