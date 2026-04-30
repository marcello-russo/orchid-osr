---
title: OrchID NCD
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

[🇮🇹 Italiano](README.it.md)

# 🌸 OrchID-NCD — Orchid Representation Learning & Novel Class Discovery

> Ultra-fine-grained visual classification and representation learning of *Ophrys* orchid species through self-supervised learning and contrastive techniques.

---

## 🎯 Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **P1 — Classification** | ResNet18/50, ConvNeXt-Tiny/Small, DINOv2+LoRA baselines, k-fold CV on clean split, per-architecture recipes, model registry | ✅ Done |
| **P1b — Experiment Explorer** | React SPA dashboard, classification upload, explainability heatmap, species stats, model selector, dataset explorer with live gallery, insight tooltips, about page | ✅ Done |
| **P2 — Embedding Space** | Supervised Contrastive Learning (SupCon) to build a structured orchid vector space | 🔜 Next |
| **P3 — Retrieval & Similarity** | Nearest-neighbor search, species clustering, unknown class detection | 📋 Planned |

---

## 🌿 Species

| | Species |
|---|---------|
| 🌸 | *Ophrys exaltata* |
| 🌺 | *Ophrys garganica* |
| 🌻 | *Ophrys incubacea* |
| 🌼 | *Ophrys majellensis* |
| 🌷 | *Ophrys sphegodes* |
| 🌱 | *Ophrys sphegodes* (Palena) |

---

## 📊 Results

Results from Serie 1 (exp6 clean split) will be published after re-running with proper flower cropping. Previous experiments (exp 1–5) were invalidated by data leakage, and exp 6 needs to be redone with a Faster R-CNN cropper for optimal flower detection.

<details>
<summary>Per-architecture training recipes</summary>

| Model | Optimizer | LR | Weight Decay | Rationale |
|-------|-----------|----|-------------|-----------|
| ResNet18/50 | Adam | 0.001 | 0 | Standard fine-tuning, BatchNorm provides implicit regularization |
| ConvNeXt-Tiny/Small | AdamW | 0.0001 | 0.05 | ConvNeXt paper recipe — LayerNorm is sensitive to large gradients |
| DINOv2-Small/Base | AdamW | 0.0002 | 0.01 | Light regularization on LoRA adapters (~0.5% trainable params) |

Shared: batch=8 (effective 32 via gradient accumulation ×4), patience=15, ReduceLROnPlateau, 100 epochs max.

</details>

---

## 🚀 Quick Start

### Prerequisites
- Linux (tested on Ubuntu/Pop!_OS)
- Podman (or Docker) + NVIDIA GPU + Container Toolkit

### Run

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/resnet18

# Start container (backend + built frontend on port 5000)
./podman-start.sh
```

### Development (with hot-reload)

```bash
# Terminal 1 — Backend
cd backend
python server.py --reload

# Terminal 2 — Frontend (Vite dev server on port 5173)
cd frontend
npm install
npm run dev
```

Frontend at http://localhost:5173, API proxy → backend :5000.

### CLI

```bash
cd backend

python cli.py --status                                          # Check status

# K-fold cross-validation on clean split (exp6)
python cli.py --cross-validate --exp 6 --model resnet18 \
  --epochs 100 --patience 15 --accumulation 4

python cli.py --cross-validate --exp 6 --model convnext_tiny \
  --epochs 100 --patience 15 --accumulation 4

# Single fold (for resume)
python cli.py --cross-validate --exp 6 --model resnet50 --fold 0 \
  --epochs 100 --patience 15 --accumulation 4

# Aggregate completed folds
python cli.py --cross-validate --exp 6 --model resnet50 \
  --epochs 100 --patience 15 --accumulation 4 --aggregate-only
```

---

## 🏗️ Project Structure

```
resnet18/
├── backend/                      # Application code
│   ├── app/                      # Source code (modular)
│   │   ├── core/                 # ML core
│   │   │   ├── models/           # Multi-model registry + classifiers (ResNet, ConvNeXt, DINOv2)
│   │   │   ├── training.py       # Training loop
│   │   │   ├── evaluation.py     # Metrics & evaluation
│   │   │   └── kfold/            # K-fold cross-validation
│   │   ├── cli/                  # CLI interface
│   │   ├── api/                  # FastAPI endpoints
│   │   │   └── routes/           # experiments, classify, species, dataset
│   │   ├── preprocessing/        # Data preprocessing
│   │   ├── explainability/       # Captum explainability
│   │   └── visualization/        # Plot generation
│   ├── experiments/              # Experiment results
│   ├── cli.py                    # CLI entrypoint
│   ├── server.py                 # API server entrypoint
│   └── requirements.txt
├── frontend/                     # React SPA
│   ├── src/
│   │   ├── components/           # Layout + Shadcn/ui components
│   │   ├── features/
│   │   │   ├── explorer/         # Charts, hooks, experiment config
│   │   │   ├── classify/         # Classification hooks
│   │   │   └── dataset/          # Dataset explorer gallery & charts
│   │   ├── pages/                # Dashboard, Experiments, Detail, Classify, Dataset, Research
│   │   ├── stores/               # Zustand (theme, sidebar)
│   │   └── lib/                  # API client, React Query, types
│   └── package.json
├── Dockerfile                    # Multi-stage (Node.js build + PyTorch)
├── podman-start.sh
├── entrypoint.sh
├── docs/
└── assets/
```

---

## 🔧 Tech Stack

| | Component |
|---|-----------|
| 🧠 | PyTorch 2.6.0 + CUDA 12.4 |
| 🏛️ | ResNet18/50, ConvNeXt-Tiny/Small, DINOv2-Small/Base + LoRA (ImageNet pretrained) |
| 🔍 | Captum (explainability) |
| 🖥️ | React 19 + TypeScript + Vite + Tailwind v4 + Shadcn/ui |
| 📊 | Recharts (interactive charts), React Query, Zustand |
| 📦 | Podman + NVIDIA Container Toolkit |

---

## 📡 API

```bash
cd backend
python server.py                    # Start API on port 5000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify/upload` | POST | Classify uploaded image (multipart, optional `model_id`) |
| `/classify/explain` | POST | Occlusion heatmap for uploaded image |
| `/classify/models` | GET | List available trained models with metrics |
| `/api/experiments` | GET | List all experiments |
| `/api/experiments/{id}` | GET | Experiment detail with folds |
| `/api/experiments/{id}/folds/{n}` | GET | Single fold data (0-based) |
| `/api/species` | GET | Species list with image counts |
| `/api/dataset/classes` | GET | Dataset classes with train/test counts |
| `/api/dataset/classes/{name}/images` | GET | Paginated image list per class |
| `/api/dataset/images/{split}/{class}/{file}` | GET | Serve dataset image |
| `/docs` | GET | Swagger UI |

---

## 📄 License

See [LICENSE](LICENSE).
