FROM docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt

COPY backend/app/ app/
COPY backend/tests/ tests/
COPY backend/cli.py .
COPY backend/server.py .

EXPOSE 5000
CMD ["python", "server.py"]
