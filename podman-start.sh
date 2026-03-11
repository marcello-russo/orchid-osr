#!/bin/bash
# Avvia il container con GPU su Podman.
# Su Docker usare: docker compose up --build -d

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
  --shm-size 16g \
  -v ./dataset:/workspace/dataset \
  -v ./models:/workspace/models \
  -v ./inference:/workspace/inference \
  -v ./app:/workspace/app \
  -v ./requirements.txt:/workspace/requirements.txt \
  -v ./notebook_clean.ipynb:/workspace/notebook_clean.ipynb \
  -v ./run_pipeline.py:/workspace/run_pipeline.py \
  -v ./run_experiments.py:/workspace/run_experiments.py \
  -v ./.env:/workspace/.env \
  -v ./logs:/workspace/logs \
  -v ./results:/workspace/results \
  -v ./experiments:/workspace/experiments \
  -p 8888:8888 \
  "$IMAGE_NAME" \
  jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''

echo "✅ Jupyter su http://localhost:8888"
