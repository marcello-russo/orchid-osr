#!/bin/bash
# Avvia il container con GPU su Podman per training e sviluppo.

CONTAINER_NAME="pytorch_app"
IMAGE_NAME="orchid-ncd-app"

docker stop "$CONTAINER_NAME" 2>/dev/null
docker rm "$CONTAINER_NAME" 2>/dev/null

echo "Build frontend..."
cd frontend && npm run build && cd ..

echo "Build image..."
docker build -t "$IMAGE_NAME" -f Dockerfile .

echo "Avvio con GPU..."
docker run -d \
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
  -v ./backend/scripts:/workspace/backend/scripts \
  -v ./.env:/workspace/backend/.env \
  -v ./frontend/dist:/workspace/backend/frontend/dist \
  -p 5000:7860 \
  "$IMAGE_NAME" \
  sleep infinity

echo "Container avviato. Comandi utili:"
echo "  docker exec pytorch_app python cli.py --status"
echo "  docker exec pytorch_app python cli.py --cross-validate --exp 6 --model resnet18 --epochs 100 --patience 15 --accumulation 4"
echo "  docker exec pytorch_app python cli.py --cross-validate --exp 6 --model resnet50 --epochs 100 --patience 15 --accumulation 4"
echo "  docker exec pytorch_app python server.py"
