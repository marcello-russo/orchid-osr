# AGENTS.md — OrchID OSR (orchid)

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
python cli.py --cross-validate --exp 1 --model resnet18   # CE baseline
python cli.py --cross-validate --exp 2 --model resnet18   # SupCon
python cli.py --cross-validate --exp 3 --model resnet18 --sphor  # SpHOR (open-world)
python cli.py --cross-validate --exp 4 --model dinov2_small --sphor  # GCD (hidden O. garganica)
python cli.py --cross-validate --exp 3 --model resnet18 --novel garganica,majellensis  # generic 2-novel override

# Single fold / aggregation
python cli.py --cross-validate --exp 1 --model resnet50 --fold 0
python cli.py --cross-validate --exp 1 --model resnet50 --aggregate-only

# NCD pipeline (Novel Class Discovery)
orchid ncd --exp 4 --model dinov2_small         # OSR + clustering (GPU)
orchid ncd --stats                                    # calibration stats
orchid ncd --delong                                   # DeLong report
orchid ncd-comparison                           # cross-model comparison table

# Embedding extraction for latent space viewer
python cli.py --embeddings --exp 3 --model dinov2_small

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

# Orchid CLI (Typer) — preferred; python cli.py is deprecated alias
orchid --help
orchid train --help
orchid train --exp 4 --model dinov2_small --sphor
orchid figures --all  # rigenera le figure in documents/_figures/ (§3.5 riproducibile)
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
./backend/scripts/smart-train.sh --exp 2       # SupCon experiments
./backend/scripts/smart-train.sh --exp 3       # SpHOR experiments
./backend/scripts/smart-train.sh --exp 4       # GCD experiments (hidden O. garganica)
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
The preliminary experiments (pre-renumbering) had undetected train/test overlap (images from same source with different crops). **Current Exp 1–4 all use the clean MD5-deduplicated split (2.192 train / 300 test, split_manifest.json, seed 57).** `dataset/train_clean` is an intermediate dir — training reads `dataset/cropped_224x224/` and `dataset/cropped_256x512/`.

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
- Sync via CLI (logic in `app/hf/`, tested): `orchid hf download|sync|upload-models|upload-ncd` (requires `HF_TOKEN`)
- Live demo: `marcellorusso-orchid-osr.hf.space`

### Database / state
No database. All state is filesystem-based: experiments JSON, model `.pt` files, dataset images. The backend API scans `experiments/` at runtime to discover available models (LRU cache, max 5 loaded at once).

### Exp 3 (SpHOR) and Exp 4 (SpHOR-NCD)
- Exp 3 adds Spherical Orthogonal Prototypes (SpHOR) on top of SupCon. Prototypes are L2-normalized hypersphere embeddings with repulsion loss.
- Exp 4 is identical to Exp 3 but excludes *O. garganica* from training (1.629 known images) — hidden novel class for OSR + GCD evaluation.
- Use `--sphor` flag for both Exp 3 and Exp 4.

### Exp 4 results (OSR + GCD)
- **OSR**: best AUROC 0.875 (DINOv2-Base + SpHOR + Ensemble), ConvNeXt-Tiny 0.850. Open-set recognition at ultra-fine granularity is achievable.
- **GCD** (Vaze et al. 2022 protocol; `orchid ncd`, SS-KM in `app/core/ncd/`): known-class recovery works (SS-KM ACC old up to 0.93), but novel-class discovery fails with the standard semi-supervised method (ACC new ~0). Only DINOv2-Base discovers the hidden species uninformed (K-means ACC new 0.83±0.04), consistent with its best OSR AUROC. Figure: `orchid figures --all` → `documents/_figures/gcd_old_new.png`. One-shots frozen in `scripts/archive/thesis-figures/`.
- GCD protocol: labeled = train embeddings of 5 known classes (1.629); pool = test (300 = 250 old + 50 garganica), L2-normalized. Metrics: ACC old/new/all (global Hungarian), NMI, ARI.

### Exp 5 (in progress, branch feat/exp5-2novel-generic-cli)
Two hidden novel classes (num_classes=4). Batch runner: `/tmp/run_exp5_all.sh` (6 models sequential, log `logs/exp5_all.log`).

### SupCon mode (exp 2)
Two-phase training: (1) contrastive pretraining with KNN validation using `SupConLoss`, (2) MLP head fine-tuning with cross-entropy on frozen backbone. Uses `app/core/kfold/` with phase-specific logic.
