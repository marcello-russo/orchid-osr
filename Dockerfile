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

# Il server non gira come root. L'entrypoint parte da root solo per sistemare i
# permessi del volume `/data` (montato root:root dal host) e poi ri-esegue sé
# stesso come `app` con setpriv: il processo che serve le richieste è `app`.
RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

WORKDIR /workspace/backend

# Wheel CPU con versione esplicita (~200 MB invece dei ~2,5 GB del wheel CUDA):
# l'immagine di produzione è CPU-only per scelta, e la versione pinnata evita
# che due build dello stesso tag installino torch diversi. Le versioni stanno
# in torch-pins.txt, lo stesso file che usa la CI: due copie divergono sempre.
COPY backend/torch-pins.txt .
# --no-deps: le dipendenze di torch (numpy, sympy, networkx, ...) arrivano
# pinnate dal lock subito dopo; qui serve solo il wheel CPU giusto.
RUN pip install --no-cache-dir --default-timeout=1000 --no-deps \
    -r torch-pins.txt \
    --index-url https://download.pytorch.org/whl/cpu

# Dipendenze pinnate (requirements.txt): l'immagine e la CI installano le stesse.
# Prima del codice, così cambiare `app/` non invalida il layer delle dipendenze.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY backend/pyproject.toml .
# Solo i dati statici. Utenti, JWT secret e DB in produzione stanno sul volume
# /data (USERS_PATH/JWT_SECRET_PATH li esporta l'entrypoint), e i merged_weights
# (1,7 GB) servono unicamente agli upload su HuggingFace: niente di tutto ciò
# deve entrare nell'immagine.
COPY backend/data/species_counts.json backend/data/kfold_indices_seed57.json data/
# `dataset` non si crea: l'entrypoint lo collega a /data/dataset. Una directory
# già presente farebbe nascere il link *dentro* (dataset/dataset).
RUN mkdir -p models
COPY backend/app/ app/
COPY backend/server.py backend/cli.py ./
COPY backend/scripts/ scripts/
# --no-deps: le dipendenze arrivano da requirements.txt; qui serve solo
# registrare il comando `orchid`. --no-build-isolation: la build usa il
# setuptools appena installato dal lock, non uno scaricato da PyPI adesso.
RUN pip install --no-cache-dir --no-deps --no-build-isolation . && \
    rm -rf build *.egg-info

COPY entrypoint.sh .
COPY --from=frontend /build/dist /workspace/frontend/dist

# Le figure PNG a 300 dpi sono il sorgente della tesi e stanno in
# frontend/public per essere versionate con il resto: la SPA serve solo i .webp,
# quindi nell'immagine i PNG non entrano (~14 MB in meno).
RUN find /workspace/frontend/dist -type f -name '*.png' -delete && \
    chmod +x entrypoint.sh && \
    chown -R app:app /workspace

ENV API_PORT=7860 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app
EXPOSE 7860

CMD ["./entrypoint.sh"]

# --- Stage 3: training GPU ----------------------------------------------------
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS training

WORKDIR /workspace/backend
ENV HOME=/workspace

COPY backend/pyproject.toml .
# Non è lo stage di produzione: qui girano CUDA 12.1 e Python 3.10, quindi le
# versioni non possono essere quelle del lock (numpy<2, torch 2.1, older libs).
# Il vincolo è tutto in constraints-training.txt, applicato alla stessa
# risoluzione di pyproject: nessun pacchetto viene declassato dopo.
COPY backend/constraints-training.txt .
RUN pip install --no-cache-dir --default-timeout=1000 . --constraint constraints-training.txt && \
    rm -rf build *.egg-info

COPY backend/data/species_counts.json backend/data/kfold_indices_seed57.json data/
COPY backend/app/ app/
COPY backend/server.py backend/cli.py ./
COPY backend/scripts/ scripts/
COPY entrypoint.sh .
# NOTA: backend/dataset/ è escluso dal contesto (.dockerignore): il dataset
# entra via volume montato o sync da hub, mai baked nell'immagine.
RUN mkdir -p models dataset && chmod +x entrypoint.sh && chown -R 42420:42420 /workspace

ENV HF_HOME=/workspace/backend/.hf_cache \
    TORCH_HOME=/workspace/backend/.torch_cache \
    APP_UID=42420
CMD ["./entrypoint.sh"]

# Lo stage finale è quello che `docker build .` produce di default.
FROM runtime AS final
