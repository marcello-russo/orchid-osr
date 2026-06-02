# OrchID NCD — HF Spaces build (CPU-only, no GPU target)
# Stage 1: Build frontend
FROM docker.io/node:22-slim AS frontend-builder

WORKDIR /workspace/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Backend + frontend (CPU-only PyTorch)
FROM docker.io/python:3.11-slim

WORKDIR /workspace/backend

RUN pip install --no-cache-dir \
    torch torchvision torcheval \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

COPY backend/scripts/ scripts/
COPY backend/data/ data/
RUN mkdir -p models dataset

COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY backend/experiments/ experiments/

COPY --from=frontend-builder /workspace/frontend/dist /workspace/backend/frontend/dist

ENV API_PORT=7860
EXPOSE 7860

CMD ["./scripts/entrypoint.sh"]

# GPU variant (for local training, override via --target=gpu)
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS gpu
WORKDIR /workspace/backend

COPY backend/pyproject.toml .
RUN pip install --no-cache-dir --default-timeout=1000 .

COPY backend/scripts/ scripts/
COPY backend/data/ data/
RUN mkdir -p models dataset

COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY backend/experiments/ experiments/

COPY --from=frontend-builder /workspace/frontend/dist /workspace/backend/frontend/dist

ENV API_PORT=7860
EXPOSE 7860

CMD ["./scripts/entrypoint.sh"]
