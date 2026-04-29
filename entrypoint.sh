#!/bin/bash
set -e

# Download clean dataset from HF if not present
if [ ! -d "dataset/train_clean" ]; then
    echo "Downloading clean dataset from HuggingFace..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'marcellorusso/orchid-ncd-dataset',
    repo_type='dataset',
    local_dir='dataset',
    allow_patterns=['train_clean/*', 'test_clean/*', 'split_manifest.json'],
)
print('Dataset downloaded')
" || echo "Dataset download failed"
fi

# Download best model weights from HF if not present
if [ ! -f "models/.models_downloaded" ]; then
    echo "Downloading model weights from HuggingFace..."
    python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    'marcellorusso/orchid-ncd-models',
    repo_type='model',
    local_dir='models',
)
# Mark as downloaded
open('models/.models_downloaded', 'w').close()
print('Models downloaded')
" || echo "Model download failed"
fi

exec python server.py
