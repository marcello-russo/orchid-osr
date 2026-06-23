# OrchID NCD — Unified Dockerfile
# Targets: cpu (HF Space, default), gpu (local), training (OVH/cloud)

# Stage 1: Build frontend (shared across all targets)
FROM docker.io/node:22-slim AS frontend-builder

WORKDIR /workspace/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2a: CPU target — HF Spaces (default)
FROM docker.io/python:3.11-slim AS cpu

WORKDIR /workspace/backend

RUN pip install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend/pyproject.toml .
RUN pip install --no-cache-dir --default-timeout=1000 .

COPY backend/scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh
COPY backend/data/ data/
RUN mkdir -p models dataset
COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .

COPY --from=frontend-builder /workspace/frontend/dist /workspace/backend/frontend/dist

ENV API_PORT=7860
EXPOSE 7860

CMD ["./scripts/entrypoint.sh"]

# Stage 2b: GPU target — local training/inference
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS gpu

WORKDIR /workspace/backend

COPY backend/pyproject.toml .
RUN pip install --no-cache-dir --default-timeout=1000 .

COPY backend/scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh
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

# Stage 2c: Training target — OVH/cloud GPU
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS training

WORKDIR /workspace/backend

COPY backend/pyproject.toml .
RUN sed -i 's/"torch",/"torch>=2.0.0,<2.2.0",/' pyproject.toml && \
    sed -i 's/"torchvision",/"torchvision>=0.15.0,<0.17.0",/' pyproject.toml && \
    pip install --no-cache-dir --default-timeout=1000 . && \
    pip install --no-cache-dir transformers==4.43.0 tokenizers==0.19.1

COPY backend/scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh
COPY backend/data/ data/
RUN mkdir -p models dataset logs && chmod -R 777 /workspace/backend
COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY backend/experiments/ experiments/

ENV HF_HOME=/workspace/backend/.hf_cache
ENV TORCH_HOME=/workspace/backend/.torch_cache
