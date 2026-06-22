#!/bin/bash
set -e

# Use persistent storage on HF Spaces (/data) — dataset, cache HF, users
if [ -d /data ]; then
    export HF_HOME=/data/.hf_cache
    if [ ! -d /data/dataset ]; then
        mkdir -p /data/dataset
    fi
    [ ! -L dataset ] && ln -sfn /data/dataset dataset 2>/dev/null || true

    # Persistent users.json (auth) — scarica/sincronizza da dataset privato
    echo "Sincronizzo users.json da dataset privato..."
        python3 -c "
from huggingface_hub import hf_hub_download
import os
try:
    path = hf_hub_download(
        repo_id='marcellorusso/orchid-ncd-dataset',
        filename='users.json',
        repo_type='dataset',
        token=os.environ.get('HF_TOKEN')
    )
    import shutil
    shutil.copy2(path, '/data/users.json')
    print('users.json scaricato')
except Exception as e:
    print(f'users.json non disponibile: {e}')
" 2>&1
    export USERS_PATH=/data/users.json
fi

# Sync experiments/models to persistent storage
if [ -d /data ]; then
    [ -d /data/experiments ] && rm -rf /data/experiments
    mkdir -p /data/experiments
    echo "Download experiment results in corso..."
    python3 << 'PYEOF'
from huggingface_hub import HfApi, hf_hub_download
import os, shutil

api = HfApi(token=os.environ.get('HF_TOKEN'))
REPO = 'marcellorusso/orchid-ncd-models'
DST = '/data/experiments'

try:
    all_items = list(api.list_repo_tree(REPO, recursive=True))
except:
    print("list_repo_tree fallito, skip")
    all_items = []

file_paths = [i.path for i in all_items if '.' in i.path.split('/')[-1]]

for src in file_paths:
    parts = src.split('/')
    if len(parts) < 2:
        continue
    if parts[0] in ('old',):
        continue
    flat = '/'.join(parts[1:])
    dst = os.path.join(DST, flat)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        downloaded = hf_hub_download(REPO, src, token=os.environ.get('HF_TOKEN'))
        shutil.copy2(downloaded, dst)
    except:
        pass

try:
    p = hf_hub_download(REPO, 'ncd_comparison.json', token=os.environ.get('HF_TOKEN'))
    shutil.copy2(p, os.path.join(DST, 'ncd_comparison.json'))
except:
    pass

print(f'Downloaded {len(file_paths)} files')
PYEOF
" 2>&1 || echo "Download experiments fallito"
    [ ! -L experiments ] && ln -sfn /data/experiments experiments 2>/dev/null || true
fi

# Download dataset if not present (uses HF_TOKEN from Secrets at runtime)
if [ ! -d "dataset/cropped_256x512/train" ] && [ ! -d "dataset/cropped_224x224/train" ]; then
    echo "Dataset non trovato — download in corso..."
    python scripts/hf_tools/download_cache.py || echo "Download fallito (forse HF_TOKEN mancante)"
fi

# Avvia server (--reload in dev mode)
EXTRA_ARGS=""
if [ "$API_RELOAD" = "true" ]; then
    EXTRA_ARGS=" --reload"
fi

exec python server.py$EXTRA_ARGS
