#!/bin/bash
set -e

# Download model weights + experiment data from HF (if not already present)
if [ ! -d "models/train_combined" ]; then
    echo "Downloading model weights from HuggingFace..."
    python3 -c "
from huggingface_hub import snapshot_download
import shutil
snapshot_download('marcellorusso/orchid-ncd-models', local_dir='hf_models')
shutil.copytree('hf_models/experiments', 'experiments', dirs_exist_ok=True)
shutil.copytree('hf_models/train_combined', 'models/train_combined', dirs_exist_ok=True)
shutil.copytree('hf_models/train_privato', 'models/train_privato', dirs_exist_ok=True)
shutil.rmtree('hf_models')
" || echo "Models download failed"
elif [ -z "$(ls -A experiments/ 2>/dev/null)" ]; then
    echo "Downloading experiment data from HuggingFace..."
    python3 -c "
from huggingface_hub import snapshot_download
import shutil
snapshot_download('marcellorusso/orchid-ncd-models', allow_patterns='experiments/*', local_dir='hf_models')
shutil.copytree('hf_models/experiments', 'experiments', dirs_exist_ok=True)
shutil.rmtree('hf_models')
" || echo "Experiment data download failed"
fi

exec python server.py
