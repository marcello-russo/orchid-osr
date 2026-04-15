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
  -v ./backend/dataset:/workspace/backend/dataset \
  -v ./backend/models:/workspace/backend/models \
  -v ./backend/experiments:/workspace/backend/experiments \
  -v ./backend/app:/workspace/backend/app \
  -v ./backend/cli.py:/workspace/backend/cli.py \
  -v ./backend/server.py:/workspace/backend/server.py \
  -v ./backend/tests:/workspace/backend/tests \
  -v ./backend/test_dataset:/workspace/backend/test_dataset \
  -v ./backend/frontend/dist:/workspace/backend/frontend/dist \
  -v ./.env:/workspace/backend/.env \
  -p 5000:5000 \
  "$IMAGE_NAME" \
  sleep infinity

echo "✅ Container avviato. Comandi utili:"
echo "   podman exec pytorch_app python -m pytest tests/ -v"
echo "   podman exec pytorch_app python cli.py --status"
echo "   podman exec pytorch_app python server.py"
echo ""
echo "   Per dev frontend (hot-reload su porta 5173):"
echo "   cd frontend && npm run dev"
