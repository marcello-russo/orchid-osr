FROM docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt

COPY app/ app/
COPY tests/ tests/
COPY cli.py .
COPY server.py .

EXPOSE 5000
CMD ["python", "server.py"]
