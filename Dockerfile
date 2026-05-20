# OrchID NCD — Dockerfile (CPU inference + HF Spaces deployment)
# For GPU training, use podman-start.sh which overrides the base image.

# Stage 1: Build frontend
FROM docker.io/node:22-slim AS frontend-builder

WORKDIR /workspace/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Backend + built frontend (CPU-only PyTorch)
FROM docker.io/python:3.11-slim

WORKDIR /workspace/backend

# Install PyTorch CPU-only (much smaller than CUDA image)
RUN pip install --no-cache-dir \
    torch torchvision torcheval \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

COPY backend/scripts/ scripts/
COPY backend/data/ data/
RUN mkdir -p models dataset

# Scarica dataset e modelli (optional — i metadati JSON sono già copiati sotto)
RUN python scripts/hf_tools/download_cache.py || true

COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY backend/experiments/ experiments/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy built frontend from stage 1
COPY --from=frontend-builder /workspace/frontend/dist /workspace/backend/frontend/dist

# HF Spaces configuration
ENV API_PORT=7860
EXPOSE 7860

CMD ["./entrypoint.sh"]

# GPU variant (override via --target=gpu during build)
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS gpu
WORKDIR /workspace/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

COPY backend/scripts/ scripts/
COPY backend/data/ data/
RUN mkdir -p models dataset

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY backend/experiments/ experiments/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

COPY --from=frontend-builder /workspace/frontend/dist /workspace/backend/frontend/dist

ENV API_PORT=7860
EXPOSE 7860

CMD ["./entrypoint.sh"]
