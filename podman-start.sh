#!/bin/bash
# Avvia il container con GPU su Podman per test e sviluppo.

CONTAINER_NAME="pytorch_app"
IMAGE_NAME="resnet18-app"

podman stop "$CONTAINER_NAME" 2>/dev/null
podman rm "$CONTAINER_NAME" 2>/dev/null

echo "📦 Build..."
podman build -t "$IMAGE_NAME" -f Dockerfile .

echo "🚀 Avvio con GPU..."
podman run -d \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  -e TZ=Europe/Rome \
  --shm-size 16g \
  -v ./dataset:/workspace/dataset \
  -v ./models:/workspace/models \
  -v ./experiments:/workspace/experiments \
  -v ./app:/workspace/app \
  -v ./cli.py:/workspace/cli.py \
  -v ./server.py:/workspace/server.py \
  -v ./tests:/workspace/tests \
  -v ./test_dataset:/workspace/test_dataset \
  -v ./.env:/workspace/.env \
  -p 5000:5000 \
  "$IMAGE_NAME" \
  sleep infinity

echo "✅ Container avviato. Comandi utili:"
echo "   podman exec pytorch_app python -m pytest tests/ -v"
echo "   podman exec pytorch_app python cli.py --status"
echo "   podman exec pytorch_app python server.py"
