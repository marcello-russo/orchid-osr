#!/bin/bash

# Configuration
IMAGE_NAME="pytorch_app_mac"
CONTAINER_NAME="pytorch_app_mac"
PORT=8888

# Check if orbstack is running (optional check, usually not needed if command works)

echo "Building and running for Mac (OrbStack)..."
# Use the mac specific compose file
docker-compose -f docker-compose.mac.yml up -d --build

echo "Container started!"
echo "Access the notebook at http://localhost:$PORT"
echo "Logs: docker-compose -f docker-compose.mac.yml logs -f"
echo ""
echo "NOTE: Running inside Docker on Mac means the code runs on Linux ARM64 CPU."
echo "Hardware MPS acceleration is currently only available when running natively on macOS,"
echo "not inside Docker containers (as of early 2026)."
echo "However, this setup is optimized for Apple Silicon (ARM64) usage."
