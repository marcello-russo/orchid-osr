# OrchidOSR — Unified Dockerfile
# Targets: cpu (HF Space, default), gpu (local), training (OVH/cloud)

# Stage: shared scripts (avoids duplicate COPY across cpu/gpu/training targets)
FROM scratch AS scripts-stage
COPY backend/scripts/ /scripts/

# Stage 2a: CPU target — HF Spaces (default)
FROM docker.io/python:3.11-slim AS cpu

WORKDIR /workspace/backend

RUN pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/pyproject.toml .
# KEEP IN SYNC with gpu/training targets
# Remove torch/torchvision to avoid double install (already installed above via CPU index)
RUN sed -i '/"torch",/d; /"torchvision",/d' pyproject.toml && \
    pip install --no-cache-dir --default-timeout=1000 .

COPY backend/data/ data/
RUN mkdir -p models dataset
COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY --from=scripts-stage /scripts/ scripts/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV API_PORT=7860
EXPOSE 7860

CMD ["./entrypoint.sh"]

# Stage 2b: GPU target — local training/inference
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS gpu

WORKDIR /workspace/backend

COPY backend/pyproject.toml .
# KEEP IN SYNC with gpu/training targets
RUN pip install --no-cache-dir --default-timeout=1000 .

COPY backend/data/ data/
RUN mkdir -p models dataset
COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY --from=scripts-stage /scripts/ scripts/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV API_PORT=7860
EXPOSE 7860

CMD ["./entrypoint.sh"]

# Stage 2c: Training target — OVH/cloud GPU
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS training

WORKDIR /workspace/backend
ENV HOME=/workspace

COPY backend/pyproject.toml .
# KEEP IN SYNC with gpu/training targets
# Relax torch pinning for OVH/cloud env
COPY backend/constraints-training.txt .
RUN     pip install --no-cache-dir --default-timeout=1000 . --constraint constraints-training.txt && \
    pip install --no-cache-dir transformers==4.43.0 tokenizers==0.19.1

COPY backend/data/ data/
COPY --from=scripts-stage /scripts/ scripts/
COPY backend/dataset/ dataset/
RUN mkdir -p models dataset
COPY backend/app/ app/
COPY backend/server.py .
COPY backend/cli.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh && chown -R 42420:42420 /workspace

ENV HF_HOME=/workspace/backend/.hf_cache
ENV TORCH_HOME=/workspace/backend/.torch_cache
CMD ["./entrypoint.sh"]

# Final stage — HF Spaces builds the last target by default.
# Declaring this ensures 'docker build .' uses the CPU target.
FROM cpu AS final
