#!/bin/bash
set -e

# Se il dataset non è stato scaricato in build (es. HF_TOKEN mancante), prova ora
if [ ! -d "dataset/cropped_256x512/train_clean" ]; then
    echo "Dataset non trovato — tentativo download all'avvio..."
    python scripts/hf_tools/download_cache.py || echo "Download dataset fallito — si usa proxy HF CDN per le immagini"
fi

exec python server.py
