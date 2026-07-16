#!/bin/bash
set -e

# Use persistent storage on HF Spaces (/data) — dataset, cache HF, users
DATA_DIR="${DATA_DIR:-/data}"
if [ -d "$DATA_DIR" ]; then
    export HF_HOME="$DATA_DIR/.hf_cache"
    if [ ! -d "$DATA_DIR/dataset" ]; then
        mkdir -p "$DATA_DIR/dataset"
    fi
    [ ! -L dataset ] && ln -sfn "$DATA_DIR/dataset" dataset 2>/dev/null || true

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
    shutil.copy2(path, \"$DATA_DIR/users.json\")
    print('users.json scaricato')
except Exception as e:
    print(f'users.json non disponibile: {e}')
" 2>&1
    export USERS_PATH="$DATA_DIR/users.json"
    # JWT secret path: persist in /data per sopravvivere ai restart
    export JWT_SECRET_PATH="$DATA_DIR/.jwt_secret"
fi

# Check frontend dist (prima di operazioni lente)
if [ ! -d "frontend/dist" ]; then
    echo "⚠️  frontend/dist non trovata. La SPA non sarà servita."
    echo "   Esegui 'cd frontend && npm run build' prima di avviare."
fi

# Pre-crea symlink per experiments (prima del server)
if [ -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR/experiments"
    [ ! -L experiments ] && ln -sfn "$DATA_DIR/experiments" experiments 2>/dev/null || true
fi

# Avvia server SUBITO
EXTRA_ARGS=""
if [ "$API_RELOAD" = "true" ]; then
    EXTRA_ARGS=" --reload"
fi

python3 server.py$EXTRA_ARGS &
SERVER_PID=$!

# Sync experiments/models in background (non blocca il server)
if [ -d "$DATA_DIR" ]; then
    (
        echo "Download experiment results..."
        PYTHONPATH=/workspace/backend python3 scripts/sync_experiments.py || echo "Experiment sync failed (non-fatal)"
    ) &
fi

# Dataset non scaricato: le immagini vengono proxyate da HuggingFace CDN al volo

wait $SERVER_PID
