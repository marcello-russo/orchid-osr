# AGENTS.md — OrchidOSR (orchid)

Master's thesis project: ultra-fine-grained classification of *Ophrys* orchids via deep learning. Python backend (FastAPI + PyTorch) + React frontend (Vite + TypeScript).

## Working directory

All code lives under `orchid-osr/`. Commands must be relative to that subtree:

- Backend: `orchid-osr/backend/`
- Frontend: `orchid-osr/frontend/`
- Docker/Podman root: `orchid-osr/`

## Prerequisites

- Python 3.10+ (conda env `orchid-mps` or `pip install ".[dev]"`)
- NVIDIA GPU + CUDA for training (CPU-only inference is supported)
- Podman/Docker + NVIDIA Container Toolkit for containerized runs

## Commands

### Backend

```bash
cd backend

# Training
python cli.py --cross-validate --exp 6 --model resnet18   # CE baseline
python cli.py --cross-validate --exp 7 --model resnet18   # SupCon
python cli.py --cross-validate --exp 8 --model resnet18 --sphor  # SpHOR (open-world)
python cli.py --cross-validate --exp 9 --model dinov2_small --sphor  # NCD (hidden majellensis)

# Single fold / aggregation
python cli.py --cross-validate --exp 6 --model resnet50 --fold 0
python cli.py --cross-validate --exp 6 --model resnet50 --aggregate-only

# NCD pipeline (Novel Class Discovery)
python cli.py --ncd --exp 9 --model dinov2_small   # OSR + clustering + figures
python cli.py --ncd-comparison                      # cross-model comparison table

# Embedding extraction for latent space viewer
python cli.py --embeddings --exp 8 --model dinov2_small

# Preprocessing pipeline
python cli.py --crop           # Faster R-CNN flower cropping
python cli.py --split          # Clean train/test split with MD5 dedup
python cli.py --resize         # Letterbox resize
python cli.py --pipeline       # All three in sequence

# API server (default :5000)
python server.py               # Use `&` for background, avoid --reload for stability

# Tests
pytest                          # all tests
pytest -m "not slow"            # skip slow explain endpoint tests
pytest tests/test_core/         # core tests only
pytest --cov=app                # with coverage
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # Vite dev server on :5173, proxies /api /classify /health → :5000
npm run build      # tsc -b && vite build
npm run lint       # eslint .
```

### Containerized

```bash
cd orchid-osr
docker compose up -d                       # Dev server (CPU, --reload, volume-mounts)
docker compose --profile gpu up -d         # GPU container (training/serve)
docker compose --profile gpu run gpu bash  # Esegui comandi nel container GPU
./backend/scripts/smart-train.sh --exp 7       # SupCon experiments
./backend/scripts/smart-train.sh --exp 8       # SpHOR experiments
./backend/scripts/smart-train.sh --exp 9       # NCD experiments (hidden majellensis)
```

## Architecture

### Key paths

```
orchid-osr/
├── backend/
│   ├── cli.py                        # CLI entry point
│   ├── server.py                     # FastAPI entry point
│   ├── app/
│   │   ├── config.py                 # pydantic-settings (reads .env)
│   │   ├── core/models/registry.py   # 6 registered models with per-architecture recipes
│   │   ├── core/models/base.py       # ClassifierModel ABC
│   │   ├── core/training.py          # train(), cross_validate(), checkpoint save/resume
│   │   ├── core/kfold/               # Stratified fold splitting, SupCon two-phase, aggregation
│   │   ├── core/losses.py            # SupConLoss
│   │   ├── preprocessing/            # Cropper (Faster R-CNN), split, resize
│   │   ├── cli/                      # Arg parsing, experiment runner, definitions
│   │   ├── api/                      # FastAPI routes + deps + schemas
│   │   └── explainability/           # Captum occlusion heatmaps
│   ├── scripts/hf_tools/             # HuggingFace upload/download
│   └── experiments/                  # Training outputs (gitignored)
├── frontend/
│   └── src/
│       ├── features/explorer/        # Experiment charts & KPIs
│       ├── features/classify/        # Image upload + classify hooks
│       ├── features/dataset/         # Dataset gallery + distribution
│       ├── pages/                    # Route pages (lazy-loaded except Dashboard/Experiments)
│       ├── locales/                  # EN + IT translations
│       └── components/               # Layout + Shadcn/ui wrappers
```

### Config sources

- **Backend config**: `app/config.py` (`pydantic-settings`, reads `.env`). Model-specific params (input_size, lr, optimizer, weight_decay) are in the model registry at `app/core/models/registry.py`, not in config.
- **Frontend env**: `frontend/.env` → `VITE_API_URL` (empty = relative paths proxied by Vite)
- **Root .env**: `orchid-osr/.env` contains `NORM_MEAN`, `NORM_STD`, and `HF_TOKEN` (do not commit)

### Image size convention

- ResNet18/50: **256×512** (letterbox, preserves aspect ratio)
- ConvNeXt, DINOv2: **224×224**
- Sizes come from the model registry, not hardcoded

## Critical gotchas

### Data leakage
Experiments 1–5 had undetected train/test overlap (images from same source with different crops). **Exp 6 (clean split) and Exp 7 (SupCon) are the valid ones.** MD5 content hashing prevents leakage. Always use `--exp 6` or `--exp 7`.

### GPU vs CPU
- Training requires CUDA (or MPS on macOS). The `smart-train.sh` script expects `/sys/class/power_supply/BAT0` for battery monitoring — it won't work on desktops/servers.
- The Dockerfile builds for **CPU-only inference** (PyTorch CPU wheels). For GPU training, use `podman-start.sh`.

### Backend has no lint/format config
No `ruff`, `mypy`, `black`, or `pre-commit` setup. Only `pytest` as a quality gate. The frontend has ESLint v9 (`npm run lint`).

### Server vs CLI
- `python server.py` starts the FastAPI API on `0.0.0.0:5000`
- `python cli.py` runs training/preprocessing offline
- The frontend dev server proxies `/api`, `/classify`, `/health` to port 5000 — both must be running for full-stack dev

### HuggingFace integration
- Dataset: `marcellorusso/orchid-osr-dataset`
- Models: `marcellorusso/orchid-osr-models`
- Download during build: `scripts/hf_tools/download_cache.py` (requires `HF_TOKEN`)
- Live demo: `marcellorusso-orchid-osr.hf.space`

### Database / state
No database. All state is filesystem-based: experiments JSON, model `.pt` files, dataset images. The backend API scans `experiments/` at runtime to discover available models (LRU cache, max 5 loaded at once).

### Exp 8 (SpHOR) and Exp 9 (NCD)
- Exp 8 adds Spherical Orthogonal Prototypes (SpHOR) on top of SupCon. Prototypes are L2-normalized hypersphere embeddings with repulsion loss.
- Exp 9 is identical to Exp 8 but excludes O. majellensis from training — left as hidden novel class for NCD evaluation.
- Use `--sphor` flag for both Exp 8 and Exp 9.

### NCD results summary
All 6 architectures in Exp 9 fail to significantly detect O. majellensis as a novel class (best AUROC = 0.497 for ConvNeXt-Tiny, near random). SpHOR does not improve open-set detection in this ultra-fine-grained domain. This is a publishable negative result.

### SupCon mode (exp 7)
Two-phase training: (1) contrastive pretraining with KNN validation using `SupConLoss`, (2) MLP head fine-tuning with cross-entropy on frozen backbone. Uses `app/core/kfold/` with phase-specific logic.
