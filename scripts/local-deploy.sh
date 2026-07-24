#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Local Deploy — OrchID"
echo "  $(date)"
echo "============================================"

# Prima esegui i check CI
cd "$(dirname "$0")/.."
bash scripts/local-ci.sh

echo ""
echo "--- Deploy: build frontend ---"
cd frontend
npm run build

echo ""
echo "--- Deploy: build backend ---"
cd ..
sudo docker compose build backend

echo ""
echo "--- Deploy: restart services ---"
sudo docker compose up -d backend nginx

echo ""
echo "============================================"
echo "  ✅ Deploy completato!"
echo "============================================"
