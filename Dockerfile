# Stage 1: Build frontend
FROM docker.io/node:22-slim AS frontend-builder

WORKDIR /workspace/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Backend + built frontend
FROM docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt

COPY backend/app/ app/
COPY backend/tests/ tests/
COPY backend/cli.py .
COPY backend/server.py .

# Copy built frontend from stage 1
COPY --from=frontend-builder /workspace/frontend/dist /workspace/backend/frontend/dist

EXPOSE 5000
CMD ["python", "server.py"]
