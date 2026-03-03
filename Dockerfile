# Development Dockerfile - fast builds, code mounted as volume
FROM docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements for dependency installation
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir --default-timeout=1000 \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio && \
    pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt

# Code is mounted via docker-compose, not copied here
