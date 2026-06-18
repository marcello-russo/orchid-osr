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
    if [ ! -d /data/experiments ]; then
        echo "Download experiment results in corso..."
        python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='marcellorusso/orchid-ncd-models',
    repo_type='model',
    token=os.environ.get('HF_TOKEN'),
    local_dir='/data/experiments',
    allow_patterns=['*/results.json', '*/config.json', '*/ncd_results.json', '*/ncd_figures/*', 'ncd_comparison.json'],
)
" 2>&1 || echo "Download experiments fallito"
    else
        echo "Experiments gia in /data"
    fi
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
