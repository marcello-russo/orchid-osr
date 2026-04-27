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

# Download new experiment results (exp6+) from HF models repo if not already present
if [ ! -d "experiments/exp6_clean_split_resnet18" ]; then
    echo "Downloading exp6+ results from HuggingFace..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'marcellorusso/orchid-ncd-models',
    local_dir='hf_models',
    allow_patterns=['serie1_ce/**', 'README.md'],
)
import shutil
src = 'hf_models/serie1_ce'
if os.path.exists(src):
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join('experiments', name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
    print('Exp6+ results downloaded')
else:
    print('No serie1_ce results found yet')
shutil.rmtree('hf_models', ignore_errors=True)
" || echo "Models download failed"
else
    echo "Exp6+ results already present, skipping download"
fi

exec python server.py
