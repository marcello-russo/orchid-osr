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

exec python server.py
