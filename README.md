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

# OrchID-NCD

> Ultra-fine-grained visual classification and Novel Class Discovery for *Ophrys* orchids.

Master's thesis project at the University of Naples Federico II (DIETI Department).

**Live demo:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)

---

## Features

- **6 deep learning architectures** — ResNet, ConvNeXt, DINOv2 with per-model optimizers and LoRA fine-tuning
- **Multi-experiment comparison** — CE baseline, SupCon contrastive learning, Spherical Orthogonal Prototypes, MGH-NCD
- **Novel Class Discovery pipeline** — Open-Set Recognition (Mahalanobis + KNN + Energy ensemble), clustering, t-SNE/UMAP visualization
- **Multi-granularity features** — concatenated activations from multiple network stages for richer representations
- **Hard negative mining** — targeted upweighting of confusing class pairs during contrastive training
- **Invite-based authentication** — admin creates invite tokens, users register with username/password, bcrypt hashing
- **Full i18n** — Italian and English UI, dynamic language switching
- **Interactive dashboard** — experiment KPIs, species distribution charts, per-fold metrics, NCD results
- **Dataset gallery** — browse and explore the 6-species dataset with train/test split views
- **Image classification** — upload or drag-and-drop, model selection dropdown, cached weights
- **Explainability** — Captum occlusion heatmaps for model interpretability
- **CI/CD pipeline** — GitHub Actions with ruff lint/format, pytest, ESLint, Docker build validation

---

## Quick Start

### Prerequisites
- Linux with NVIDIA GPU + Docker

### Run

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/orchid-ncd
./docker-start.sh
```

### Development

```bash
# Backend
cd backend && python server.py --reload

# Frontend
cd frontend && npm install && npm run dev
```

---

## API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/login` | POST | — | Login |
| `/auth/register` | POST | — | Register with invite code |
| `/classify` | POST | Public | Classify image (base64) |
| `/classify/upload` | POST | Public | Classify uploaded image |
| `/classify/explain` | POST | User | Captum heatmap |
| `/classify/models` | GET | Public | Available models |
| `/api/experiments` | GET | Public | All experiments |
| `/api/species` | GET | Public | Species with counts |
| `/api/comparison` | GET | Public | Cross-experiment comparison |
| `/api/dataset/classes` | GET | User | Class counts |
| `/api/embeddings` | GET | User | Latent space (t-SNE/UMAP) |
| `/api/ncd/{model}/results` | GET | User | NCD results |
| `/auth/me` | GET | Bearer | Current user |
| `/docs` | GET | — | Swagger UI |

---

## Architecture

| Layer | Technology |
|-------|-----------|
| Training | PyTorch + CUDA, torchvision, Captum, LoRA (PEFT) |
| Backend | FastAPI, Python 3.12, JWT auth, bcrypt |
| Frontend | React 19, TypeScript, Vite, Tailwind v4, Shadcn/ui, Recharts |
| Infra | Docker, GitHub Actions, HuggingFace Spaces |

---

## Dataset

Six cryptic *Ophrys* species, private field photos + internet images. Available on HuggingFace:

```python
from huggingface_hub import snapshot_download
snapshot_download("marcellorusso/orchid-ncd-dataset", repo_type="dataset", local_dir="dataset")
```

## Links

- **Live demo:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)
- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID)
- **Models:** `marcellorusso/orchid-ncd-models`
- **Dataset:** `marcellorusso/orchid-ncd-dataset`

## License

[MIT](LICENSE)
