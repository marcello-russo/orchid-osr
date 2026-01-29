FROM docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

COPY requirements.txt /workspace/
COPY app /workspace/app/
COPY .env /workspace/
COPY notebook.ipynb /workspace/

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=1000 \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio && \
    pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt

EXPOSE 8888 5000

#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
