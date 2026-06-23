#!/bin/bash
# Sync local experiments from HuggingFace models repo.
# Downloads missing config.json, results.json, fold_*.json, best_fold_N.pt
# for experiment directories that lack these metadata files.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_DIR="$BACKEND_DIR/experiments"

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    if [ -f "$BACKEND_DIR/../.env" ]; then
        source <(grep HF_TOKEN "$BACKEND_DIR/../.env" | sed 's/ //g')
    fi
fi
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set and not found in .env"
    exit 1
fi

echo "Syncing experiments from marcellorusso/orchid-ncd-models..."
echo "Experiment dir: $EXP_DIR"

python3 << 'PYEOF'
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

TOKEN = os.environ.get('HF_TOKEN', '')
if not TOKEN:
    print("HF_TOKEN not available")
    sys.exit(1)

api = HfApi(token=TOKEN)
REPO = 'marcellorusso/orchid-ncd-models'
EXP_DIR = Path(os.environ.get('EXP_DIR', '/workspace/backend/experiments'))

# List all files on HF
print("Listing HF repo contents...")
try:
    items = list(api.list_repo_tree(REPO, recursive=True))
except Exception as e:
    print(f"Failed to list repo: {e}")
    sys.exit(1)

hf_paths = [i.path for i in items if isinstance(i, RepoFile)]
print(f"Found {len(hf_paths)} files on HF")

# Build set of exp dirs on HF (e.g., exp8_sphor_resnet18)
hf_exps = {}
for p in hf_paths:
    parts = p.split('/')
    if len(parts) >= 2 and parts[0] not in ('old',):
        exp_name = parts[1]  # full name like exp8_sphor_resnet18
        if exp_name not in hf_exps:
            hf_exps[exp_name] = []
        hf_exps[exp_name].append(p)

print(f"Found {len(hf_exps)} experiment dirs on HF")

# Scan local experiment dirs
local_dirs = sorted([d for d in EXP_DIR.iterdir() if d.is_dir() and d.name.startswith('exp')])
synced = 0
skipped = 0

for d in local_dirs:
    name = d.name
    has_config = (d / 'config.json').exists()
    has_results = (d / 'results.json').exists()
    has_best_pt = any(f.name.startswith('best_fold_') and f.name.endswith('.pt') for f in d.iterdir())

    needs_config = not has_config
    needs_results = not has_results
    needs_best = not has_best_pt

    if not needs_config and not needs_results and not needs_best:
        continue  # already complete

    if name not in hf_exps:
        print(f"  SKIP {name}: not found on HF")
        skipped += 1
        continue

    print(f"  {name}: missing config={needs_config} results={needs_results} best_pt={needs_best}")

    for hf_path in hf_exps[name]:
        filename = hf_path.split('/')[-1]
        
        # Skip files we don't need
        if filename.startswith('REPORT_'):
            continue
        
        # Only download what's actually missing
        if filename == 'config.json' and not needs_config:
            continue
        if filename == 'results.json' and not needs_results:
            continue
        if filename.startswith('best_fold_') and filename.endswith('.pt') and not needs_best:
            continue
        if filename.startswith('fold_') and filename.endswith('.json') and not needs_results:
            continue
        
        # Skip adapter directories, logs, embeddings (handled separately)
        if '/' in hf_path and len(hf_path.split('/')) > 2:
            sub = hf_path.split('/', 2)[2]
            if sub.startswith('best_fold_') and '_adapter' in sub:
                continue
            if sub.startswith('embeddings'):
                continue
        
        dst = d / filename
        if dst.exists():
            continue
        try:
            dl = hf_hub_download(REPO, hf_path, token=TOKEN)
            import shutil
            shutil.copy2(dl, dst)
            print(f"    Downloaded {filename}")
            synced += 1
        except Exception as e:
            print(f"    FAILED {filename}: {e}")

print(f"\nSynced {synced} files, skipped {skipped} experiments")
PYEOF
