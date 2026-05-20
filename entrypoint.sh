#!/bin/bash
set -e

# Se i modelli non sono stati scaricati in build (es. HF_TOKEN mancante), prova ora
if [ ! -f "experiments/.models_synced" ]; then
    echo "Modelli non trovati — tentativo download all'avvio..."
    python scripts/hf_tools/download_cache.py || echo "Download fallito — alcuni esperimenti potrebbero non essere disponibili"
fi

exec python server.py
