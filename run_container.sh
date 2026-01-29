#!/bin/bash

# Configuration
IMAGE_NAME="localhost/resnet18-app:latest"
CONTAINER_NAME="pytorch_app"
PORT=8888

# Check if image exists, otherwise build it
if ! podman image exists $IMAGE_NAME; then
    echo "Image $IMAGE_NAME not found. Building..."
    podman build -t $IMAGE_NAME .
fi

# Remove existing container if it exists
podman rm -f $CONTAINER_NAME 2>/dev/null

# Run the container with NVIDIA GPU support (CDI)
echo "Starting container $CONTAINER_NAME with GPU support..."
podman run -d \
    --name $CONTAINER_NAME \
    --device nvidia.com/gpu=all \
    --shm-size=16g \
    -p $PORT:8888 \
    -v $(pwd)/dataset:/workspace/dataset \
    -v $(pwd)/models:/workspace/models \
    -v $(pwd)/inference:/workspace/inference \
    -v $(pwd)/app:/workspace/app \
    -v $(pwd)/requirements.txt:/workspace/requirements.txt \
    -v $(pwd)/notebook.ipynb:/workspace/notebook.ipynb \
    -v $(pwd)/.env:/workspace/.env \
    -v $(pwd)/logs:/workspace/logs \
    -v $(pwd)/results:/workspace/results \
    $IMAGE_NAME \
    jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''

echo "Container started!"
echo "You can access the notebook at http://localhost:$PORT"
echo "Logs are available via: podman logs -f $CONTAINER_NAME"
