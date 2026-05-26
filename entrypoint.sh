#!/bin/bash
set -e

# Download dataset if not present (uses HF_TOKEN from Secrets at runtime)
if [ ! -d "dataset/cropped_256x512/train_clean" ] && [ ! -d "dataset/cropped_224x224/train_clean" ]; then
    echo "Dataset non trovato — download in corso..."
    python scripts/hf_tools/download_cache.py || echo "Download fallito (forse HF_TOKEN mancante)"
fi

exec python server.py
