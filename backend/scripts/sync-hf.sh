#!/bin/bash
# Sync current repo to HF Space
# Usage: bash scripts/sync-hf.sh [--force]

set -e
cd "$(dirname "$0")/../.."

HF_TOKEN=$(grep HF_TOKEN .env 2>/dev/null | cut -d= -f2)
if [ -z "$HF_TOKEN" ]; then
  echo "❌ HF_TOKEN non trovato in .env"
  exit 1
fi

echo "🔄 Syncing to HF Space..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_folder(
    folder_path='.',
    repo_id='marcellorusso/orchid-ncd',
    repo_type='space',
    commit_message='Sync from local',
    ignore_patterns=[
        '.git/*', 'backend/dataset/*', 'backend/models/*',
        'backend/experiments/*', '*.pt', '*.safetensors',
        'backend/tests/*', 'node_modules/*', 'frontend/node_modules/*',
        'docs/*', 'assets/*', '*.npy', '*.png', '*.jpg', '*.webp',
        '.env', '.gitignore', '*.log', 'smart-train.log',
    ],
)
print('✅ HF Space sync complete')
" && curl -s -X POST "https://huggingface.co/api/spaces/marcellorusso/orchid-ncd/restart" \
  -H "Authorization: Bearer $HF_TOKEN" 2>/dev/null | python3 -c "import json,sys; print('🚀 Space:', json.load(sys.stdin).get('stage','?'))"