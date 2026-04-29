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

# Download model weights from HF into experiments/ so backend can find them
if [ ! -f "experiments/.models_synced" ]; then
    echo "Downloading model weights from HuggingFace..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
# Download serie1_ce models — .pt files go into experiments/ directly
snapshot_download(
    'marcellorusso/orchid-ncd-models',
    repo_type='model',
    local_dir='experiments',
    allow_patterns=['serie1_ce/**/*.pt', 'serie1_ce/**/*.json', 'serie1_ce/**/*.safetensors'],
)
# Flatten: move serie1_ce/exp6_*/ to experiments/exp6_*/
src = 'experiments/serie1_ce'
if os.path.isdir(src):
    for d in os.listdir(src):
        s = os.path.join(src, d)
        t = os.path.join('experiments', d)
        if os.path.isdir(s) and not os.path.exists(t):
            os.rename(s, t)
            print(f'Moved {d}')
    os.rmdir(src)
open('experiments/.models_synced', 'w').close()
print('Models synced')
" || echo "Model download failed"
fi

exec python server.py
