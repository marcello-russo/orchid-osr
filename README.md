---
title: OrchID NCD
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

[🇮🇹 Italiano](README.it.md)

# 🌸 OrchID — Orchid Representation Learning

> Learning structured vector spaces of *Ophrys* orchid species through contrastive learning and metric embedding techniques.

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

## 📊 Results — Serie 1 (Clean Split, CE Loss)

5-fold StratifiedKFold on exp6 clean split (seed=57). 2,017 train / 300 test (50/class, balanced). Zero data leakage verified via MD5.

| Model | Val F1 Macro | Test Accuracy | Test F1 Macro | Status |
|-------|-------------|---------------|---------------|--------|
| ResNet18 | 0.868 ± 0.016 | 85.00% | 84.96% | ✅ |
| ResNet50 | 0.905 ± 0.027 | 88.00% | 88.01% | ✅ |
| ConvNeXt-Tiny | 0.862 ± 0.020 | 84.00% | 83.91% | ✅ |
| DINOv2-Small + LoRA | 0.893 ± 0.006 | 85.33% | 85.50% | ✅ |
| DINOv2-Base + LoRA | — | — | — | 🔜 Training |
| ConvNeXt-Small | — | — | — | 📋 Pending |

<details>
<summary>Per-architecture training recipes</summary>

| Model | Optimizer | LR | Weight Decay | Rationale |
|-------|-----------|----|-------------|-----------|
| ResNet18/50 | Adam | 0.001 | 0 | Standard fine-tuning, BatchNorm provides implicit regularization |
| ConvNeXt-Tiny/Small | AdamW | 0.0001 | 0.05 | ConvNeXt paper recipe — LayerNorm is sensitive to large gradients |
| DINOv2-Small/Base | AdamW | 0.0002 | 0.01 | Light regularization on LoRA adapters (~0.5% trainable params) |

Shared: batch=8 (effective 32 via gradient accumulation ×4), patience=15, ReduceLROnPlateau, 100 epochs max.

</details>

<details>
<summary>Legacy experiments (exp1-5, data leakage)</summary>

These experiments used combined datasets with content overlap between train and test (verified via MD5). Results are inflated and not reliable.

| Experiment | Train | Test | Accuracy | F1 Macro | Notes |
|------------|-------|------|----------|----------|-------|
| Exp1 | Private | Private | 94.78% | 93.72% | Baseline (6 classes) |
| Exp2 | Private | Internet Raw | 35.09% | 37.57% | Strong domain shift |
| Exp3 | Private | Internet Cropped | 48.15% | 50.43% | Cropping helps (+12.9%) |
| Exp4 | Combined | Internet Cropped | 95.19% | 95.37% | Data leakage present |
| Exp5 | Combined | Private | 96.79% | 95.83% | Data leakage present |

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
  --epochs 100 --patience 15 --overfit-threshold 0 --accumulation 4

python cli.py --cross-validate --exp 6 --model convnext_tiny \
  --epochs 100 --patience 15 --overfit-threshold 0 --accumulation 4

# Single fold (for resume)
python cli.py --cross-validate --exp 6 --model resnet50 --fold 0 \
  --epochs 100 --patience 15 --overfit-threshold 0 --accumulation 4

# Aggregate completed folds
python cli.py --cross-validate --exp 6 --model resnet50 \
  --epochs 100 --patience 15 --overfit-threshold 0 --accumulation 4 --aggregate-only
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
│   │   ├── pages/                # Dashboard, Experiments, Detail, Classify, Dataset, About
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
