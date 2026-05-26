#!/bin/bash
set -e

# Use persistent storage on HF Spaces (/data) — dataset e cache HF tra rebuild
if [ -d /data ]; then
    export HF_HOME=/data/.hf_cache
    if [ ! -d /data/dataset ]; then
        mkdir -p /data/dataset
    fi
    [ ! -L dataset ] && ln -sfn /data/dataset dataset 2>/dev/null || true
fi

# Download dataset if not present (uses HF_TOKEN from Secrets at runtime)
if [ ! -d "dataset/cropped_256x512/train_clean" ] && [ ! -d "dataset/cropped_224x224/train_clean" ]; then
    echo "Dataset non trovato — download in corso..."
    python scripts/hf_tools/download_cache.py || echo "Download fallito (forse HF_TOKEN mancante)"
fi

exec python server.py
