# OrchID OSR — immagini del progetto
#
#   runtime  (default) — API + SPA inclusa, CPU-only, è l'immagine di produzione
#   training           — CUDA, solo per addestrare su GPU (mai in produzione)
#
# La SPA viene compilata in uno stage node e copiata dentro l'immagine runtime:
# in produzione non esiste un build sul server né un bind mount della dist.

# --- Stage 1: build della SPA -------------------------------------------------
FROM docker.io/node:22-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: runtime CPU (default, produzione) -------------------------------
FROM docker.io/python:3.11-slim AS runtime

WORKDIR /workspace/backend

RUN pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/pyproject.toml .
# KEEP IN SYNC con lo stage training
# torch/torchvision già installati sopra dall'indice CPU: evitiamo il doppio download
RUN sed -i '/"torch",/d; /"torchvision",/d' pyproject.toml && \
    pip install --no-cache-dir --default-timeout=1000 .

COPY backend/data/ data/
RUN mkdir -p models dataset
COPY backend/app/ app/
COPY backend/server.py backend/cli.py ./
COPY backend/scripts/ scripts/
COPY entrypoint.sh .
COPY --from=frontend /build/dist /workspace/frontend/dist
RUN chmod +x entrypoint.sh

ENV API_PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["./entrypoint.sh"]

# --- Stage 3: training GPU ----------------------------------------------------
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS training

WORKDIR /workspace/backend
ENV HOME=/workspace

COPY backend/pyproject.toml .
# KEEP IN SYNC con lo stage runtime
# Su cloud il pinning di torch viene rilassato: vince il constraints file.
COPY backend/constraints-training.txt .
RUN pip install --no-cache-dir --default-timeout=1000 . --constraint constraints-training.txt && \
    pip install --no-cache-dir transformers==4.43.0 tokenizers==0.19.1

COPY backend/data/ data/
COPY backend/app/ app/
COPY backend/server.py backend/cli.py ./
COPY backend/scripts/ scripts/
COPY entrypoint.sh .
# NOTA: backend/dataset/ è escluso dal contesto (.dockerignore): il dataset
# entra via volume montato o sync da hub, mai baked nell'immagine.
RUN mkdir -p models dataset && chmod +x entrypoint.sh && chown -R 42420:42420 /workspace

ENV HF_HOME=/workspace/backend/.hf_cache \
    TORCH_HOME=/workspace/backend/.torch_cache
CMD ["./entrypoint.sh"]

# Lo stage finale è quello che `docker build .` produce di default.
FROM runtime AS final
