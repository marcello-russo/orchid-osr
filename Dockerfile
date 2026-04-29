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
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY backend/scripts/ scripts/
RUN mkdir -p models dataset
COPY backend/experiments/ experiments/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy built frontend from stage 1
COPY --from=frontend-builder /workspace/frontend/dist /workspace/backend/frontend/dist

# HF Spaces configuration
ENV API_PORT=7860
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
EXPOSE 7860

CMD ["./entrypoint.sh"]
