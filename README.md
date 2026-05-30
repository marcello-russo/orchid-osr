---
title: OrchID NCD
emoji: "\U0001F338"
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

[Italiano](README.it.md)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/TypeScript-5.7-3178C6?style=flat&logo=typescript&logoColor=white" alt="TypeScript"/>
  <img src="https://img.shields.io/badge/Vite-6-646CFF?style=flat&logo=vite&logoColor=white" alt="Vite"/>
  <img src="https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?style=flat&logo=tailwindcss&logoColor=white" alt="Tailwind"/>
  <br/>
  <img src="https://img.shields.io/badge/ResNet-18|50-2496ED?style=flat&logo=pytorch&logoColor=white" alt="ResNet"/>
  <img src="https://img.shields.io/badge/ConvNeXt-Tiny|Small-FF6F00?style=flat&logo=pytorch&logoColor=white" alt="ConvNeXt"/>
  <img src="https://img.shields.io/badge/DINOv2-Small|Base-8B5CF6?style=flat&logo=pytorch&logoColor=white" alt="DINOv2"/>
  <br/>
  <img src="https://img.shields.io/badge/HuggingFace-Spaces|Datasets-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white" alt="GitHub Actions"/>
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA"/>
</p>

# OrchID-NCD — Orchid Representation Learning & Novel Class Discovery

> Ultra-fine-grained visual classification and representation learning of *Ophrys* orchid species through self-supervised learning and contrastive techniques.

Master's thesis project at the University of Naples Federico II (DIETI Department).

**Live demo:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)

---

## The Ophrys Problem

The genus *Ophrys* is one of the most taxonomically challenging orchid groups in the Mediterranean. These orchids use sexual deception to attract species-specific pollinators — flowers mimic female insects through visual, olfactory, and tactile signals. This extreme specialization drives rapid morphological divergence and frequent hybridization, producing visually ambiguous specimens that challenge even expert botanists.

The *Ophrys sphegodes* complex alone encompasses multiple cryptic taxa with overlapping phenotypic traits, making it an ideal testbed for ultra-fine-grained classification and novel class discovery.

---

## Species

Six taxa from the *Ophrys sphegodes* complex:

| Taxon | Images | Source |
|-------|--------|--------|
| *O. exaltata* | 501 train / 50 test | private + internet |
| *O. garganica* | 570 train / 50 test | private + internet |
| *O. incubacea* | 366 train / 50 test | private + internet |
| *O. majellensis* | 152 train / 50 test | private only |
| *O. sphegodes* | 433 train / 50 test | private + internet |
| *O. sphegodes* (Palena) | 210 train / 50 test | private only |
| **Total** | **2232 / 300** | |

---

## Pipeline

The preprocessing pipeline is a modular CLI — each step is independent and resumable:

```
Raw images (private + internet)
        |
        v
   [1] CROP          --crop
   Faster R-CNN flower detection
        |
        v
   [2] SPLIT         --split
   Clean split + MD5 dedup
        |
        v
   [3] RESIZE        --resize
   Letterbox per-architecture
        |
        v
   [4] TRAIN         --cross-validate --exp 10
   5-fold stratified CV
```

**Key design decisions:**
- **Flower cropping:** Faster R-CNN (ResNet50-FPN) detects and crops flowers from full-resolution photos
- **Zero data leakage:** MD5 content hashing ensures no overlap between train and test
- **Letterbox resizing:** Per-architecture sizing (ResNet 256×512, ConvNeXt/DINOv2 224×224)

---

## Experiments

| Exp | Method | Description | Status |
|-----|--------|-------------|--------|
| 6 | CE Baseline | Cross-entropy on clean split, 6 architectures | ✅ Done |
| 7 | SupCon | Supervised Contrastive Learning + CE head | ✅ Done |
| 8 | SpHOR | SupCon + Spherical Orthogonal Prototypes | ✅ Done |
| 9 | NCD | Exp 8 without O. majellensis (hidden novel class) | ✅ Done |
| 10 | **MGH-NCD** | Multi-Granularity Hard-negative NCD | ✅ Done |

### Exp 10 — MGH-NCD

Multi-Granularity Hard-negative Novel Class Discovery:
- **Multi-granularity features:** concatenates outputs from multiple network stages (e.g., stage3 + stage4)
- **Hard negative mining:** 5× weight on (majellensis, sphegodes) and 3× on (sphegodes, Palena) pairs
- **Full fine-tune:** backbone unfrozen in Phase 2 for non-ViT models
- **LoRA r=32:** applied in Phase 1 for parameter-efficient adaptation of ViT backbones
- **OSR Ensemble:** Mahalanobis + KNN + Energy scores for open-set detection

---

## Architecture Comparison

| Family | Models | Why |
|--------|--------|-----|
| **ResNet** | ResNet18, ResNet50 | Hierarchical local features via residual connections |
| **ConvNeXt** | ConvNeXt-Tiny, ConvNeXt-Small | Modernized CNN with large kernels and LayerNorm |
| **DINOv2** | DINOv2-Small, DINOv2-Base | Self-supervised ViT with global attention, fine-tuned via LoRA |

<details>
<summary>Per-architecture training recipes</summary>

| Model | Optimizer | LR | Weight Decay |
|-------|-----------|----|-------------|
| ResNet18/50 | Adam | 0.001 | 0 |
| ConvNeXt-Tiny/Small | AdamW | 0.0003 | 0.05 |
| DINOv2-Small/Base | AdamW | 0.0002 | 0.01 |

Shared: 5-fold stratified CV, balanced augmentation per-fold, patience=15.

</details>

---

## Results

### Exp 10 — MGH-NCD (Test Set)

| Model | Accuracy | F1 Macro | Best Fold |
|-------|----------|----------|-----------|
| ResNet18 | — | — | fold 4 |
| ResNet50 | 0.35 | 0.32 | fold 3 |
| ConvNeXt-Tiny | — | — | fold 2 |
| ConvNeXt-Small | — | — | fold 3 |
| DINOv2-Small | — | — | fold 4 |
| DINOv2-Base | 0.73 | 0.70 | fold 1 |

### O. majellensis Detection

| Model | Recall | F1 |
|-------|--------|----|
| DINOv2-Base | 0.16 | 0.28 |
| ResNet50 | 0.00 | 0.00 |

*Majellensis remains the hardest class — visually near-identical to sphegodes.*

---

## Authentication

The app supports invite-based registration with three roles:

| Role | Access |
|------|--------|
| **Public** | Health, classify, experiments, species, comparison, NCD figures |
| **Registered** | Everything public + dataset classes, explain, embeddings, NCD results |
| **Admin** | Everything + dataset images, cache management |

### Setup

```bash
cd backend

# Generate invite tokens
python cli.py --gen-invites 5

# Create user directly
python cli.py --add-user username --password pass --role user

# List users and invites
python cli.py --list-users
```

Passwords are hashed with bcrypt and stored in `data/users.json` (gitignored). On HF Spaces, the file is persisted in `/data/` and synced to the private dataset repo.

---

## Dataset

Available on HuggingFace: `marcellorusso/orchid-ncd-dataset` (private)

```python
from huggingface_hub import snapshot_download
snapshot_download("marcellorusso/orchid-ncd-dataset", repo_type="dataset", local_dir="dataset")
```

---

## Quick Start

### Prerequisites
- Linux (tested on Ubuntu/Pop!_OS)
- Docker (or Docker) + NVIDIA GPU + Container Toolkit

### Run

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/orchid-ncd
./docker-start.sh
```

### Development

```bash
# Terminal 1 — Backend
cd backend
python server.py --reload

# Terminal 2 — Frontend (Vite dev server on :5175)
cd frontend
npm install && npm run dev
```

### CLI

```bash
cd backend

# Preprocessing
python cli.py --crop --split --resize --pipeline

# Training
python cli.py --cross-validate --exp 10 --model dinov2_small --full-finetune --hard-weight 5.0

# Smart trainer (all models, battery-aware)
./smart-train.sh --exp 10
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Training | PyTorch + CUDA, torchvision, Captum |
| Models | ResNet18/50, ConvNeXt-Tiny/Small, DINOv2-Small/Base |
| Backend | FastAPI, Python 3.12 |
| Frontend | React 19, TypeScript, Vite, Tailwind v4, Shadcn/ui, Recharts |
| Infra | Docker + NVIDIA Container Toolkit, HuggingFace Spaces |

---

## API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/register` | POST | — | Register with invite code |
| `/auth/login` | POST | — | Login, returns JWT |
| `/auth/me` | GET | Bearer | Current user info |
| `/classify/upload` | POST | Public | Classify uploaded image |
| `/classify/explain` | POST | User | Occlusion heatmap |
| `/classify/models` | GET | Public | Available models |
| `/api/experiments` | GET | Public | List experiments |
| `/api/species` | GET | Public | Species with counts |
| `/api/dataset/classes` | GET | User | Dataset class counts |
| `/api/dataset/images/{split}/{class}/{file}` | GET | Admin | Serve dataset image |
| `/docs` | GET | — | Swagger UI |

---

## Links

- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID)
- **Live demo:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)
- **Models:** `marcellorusso/orchid-ncd-models`
- **Dataset:** `marcellorusso/orchid-ncd-dataset`

## License

[MIT](LICENSE)
