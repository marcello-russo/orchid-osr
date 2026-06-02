#!/bin/bash
# Avvia il container con GPU per training e sviluppo.
# Usage: ./backend/scripts/docker-start.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

CONTAINER_NAME="pytorch_app"
IMAGE_NAME="orchid-ncd-app"

docker stop "$CONTAINER_NAME" 2>/dev/null
docker rm "$CONTAINER_NAME" 2>/dev/null

echo "Build frontend..."
cd "$ROOT_DIR/frontend" && npm run build && cd "$ROOT_DIR"

echo "Build image (GPU target)..."
docker build --target gpu -t "$IMAGE_NAME" -f "$ROOT_DIR/Dockerfile" "$ROOT_DIR"

echo "Avvio con GPU..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  -e TZ=Europe/Rome \
  --shm-size 16g \
  -v "$ROOT_DIR/backend/dataset:/workspace/backend/dataset" \
  -v "$ROOT_DIR/backend/models:/workspace/backend/models" \
  -v "$ROOT_DIR/backend/experiments:/workspace/backend/experiments" \
  -v "$ROOT_DIR/backend/app:/workspace/backend/app" \
  -v "$ROOT_DIR/backend/cli.py:/workspace/backend/cli.py" \
  -v "$ROOT_DIR/backend/server.py:/workspace/backend/server.py" \
  -v "$ROOT_DIR/backend/tests:/workspace/backend/tests" \
  -v "$ROOT_DIR/backend/scripts:/workspace/backend/scripts" \
  -v "$ROOT_DIR/.env:/workspace/backend/.env" \
  -v "$ROOT_DIR/frontend/dist:/workspace/backend/frontend/dist" \
  -p 5000:7860 \
  "$IMAGE_NAME" \
  sleep infinity

echo "Container avviato. Comandi utili:"
echo "  docker exec pytorch_app python cli.py --status"
echo "  docker exec pytorch_app python cli.py --cross-validate --exp 6 --model resnet18 --epochs 100 --patience 15 --accumulation 4"
