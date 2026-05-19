---
title: OrchID NCD
emoji: "\U0001F338"
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

[Italiano](README.it.md)

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
   [4] TRAIN         --cross-validate --exp 6
   5-fold stratified CV
```

**Key design decisions:**

- **Flower cropping:** Faster R-CNN (ResNet50-FPN) detects and crops flowers from full-resolution photos, focusing the model on discriminative floral structures
- **Zero data leakage:** MD5 content hashing ensures no overlap between train and test — previous experiments (exp 1–5) were invalidated when this was discovered
- **Letterbox resizing:** Images are scaled to fit within architecture-specific dimensions while preserving aspect ratio, with black padding filling the remaining space
- **Per-architecture sizing:** ResNet uses 256x512, ConvNeXt and DINOv2 use 224x224 — sizes come from the model registry, not hardcoded

---

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **P1 — Classification** | 6 architectures, k-fold CV, per-architecture recipes, Faster R-CNN cropper pipeline | Done |
| **P1b — Explorer** | React SPA dashboard, upload classification, explainability, dataset gallery, i18n IT/EN | Done |
| **P2 — Embedding Space** | Supervised Contrastive Learning (SupCon) to structure the latent space | Next |
| **P3 — Novel Class Discovery** | Nearest-neighbor retrieval, species clustering, open-set recognition | Planned |

---

## Architecture Comparison

Three families of architectures with different inductive biases:

| Family | Models | Why |
|--------|--------|-----|
| **ResNet** | ResNet18, ResNet50 | Hierarchical local features via residual connections — proven fine-tuning baseline |
| **ConvNeXt** | ConvNeXt-Tiny, ConvNeXt-Small | Modernized CNN with large kernels and LayerNorm — stronger inductive bias for texture |
| **DINOv2** | DINOv2-Small, DINOv2-Base | Self-supervised ViT with global attention, trained on 142M images — fine-tuned via LoRA adapters (~0.5% trainable params) |

<details>
<summary>Per-architecture training recipes</summary>

| Model | Optimizer | LR | Weight Decay | Rationale |
|-------|-----------|----|-------------|-----------|
| ResNet18/50 | Adam | 0.001 | 0 | BatchNorm provides implicit regularization |
| ConvNeXt-Tiny/Small | AdamW | 0.0001 | 0.05 | ConvNeXt paper recipe — LayerNorm sensitive to large gradients |
| DINOv2-Small/Base | AdamW | 0.0002 | 0.01 | Light regularization on LoRA adapters |

Shared: physical batch=8 (effective 32 via gradient accumulation), patience=15 on val_loss, 5-fold stratified CV, balanced augmentation per-fold.

</details>

---

## Dataset

Available on HuggingFace: `marcellorusso/orchid-ncd-dataset` (private)

```
raw/
  private/                    Original field photos (6 classes, 1252 images)
  internet/                   Internet-sourced images (4 classes, 1080 images)
cropped/
  256x512/                    Letterbox-resized for ResNet
    train/  test/
  224x224/                    Letterbox-resized for ConvNeXt / DINOv2
    train/  test/
```

```python
from huggingface_hub import snapshot_download
snapshot_download("marcellorusso/orchid-ncd-dataset", repo_type="dataset", local_dir="dataset")
```

---

## Quick Start

### Prerequisites
- Linux (tested on Ubuntu/Pop!_OS)
- Podman (or Docker) + NVIDIA GPU + Container Toolkit

### Run

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/resnet18
./podman-start.sh
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

# Preprocessing pipeline (individual steps)
python cli.py --crop                    # Faster R-CNN flower cropping
python cli.py --split                   # Clean train/test split
python cli.py --resize                  # Letterbox resize for all architectures
python cli.py --pipeline                # All three in sequence

# Training
python cli.py --cross-validate --exp 6 --model resnet18
python cli.py --cross-validate --exp 6 --model convnext_tiny
python cli.py --cross-validate --exp 6 --model dinov2_base

# Single fold / aggregation
python cli.py --cross-validate --exp 6 --model resnet50 --fold 0
python cli.py --cross-validate --exp 6 --model resnet50 --aggregate-only
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Training | PyTorch 2.6 + CUDA 12.4, torchvision, Captum |
| Models | ResNet18/50, ConvNeXt-Tiny/Small, DINOv2-Small/Base + LoRA |
| Backend | FastAPI, Python 3.12 |
| Frontend | React 19, TypeScript, Vite, Tailwind v4, Shadcn/ui, Recharts |
| Infra | Podman + NVIDIA Container Toolkit, HuggingFace Spaces (Docker SDK) |

---

## Project Structure

```
resnet18/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── models/           Multi-model registry + classifiers
│   │   │   ├── kfold/            5-fold stratified cross-validation
│   │   │   ├── training.py       Training loop with gradient accumulation
│   │   │   └── evaluation.py     Metrics & evaluation
│   │   ├── preprocessing/        Cropper (Faster R-CNN), split, letterbox resize
│   │   ├── cli/                  CLI interface (experiments, runner, definitions)
│   │   ├── api/                  FastAPI endpoints
│   │   ├── explainability/       Captum explainability
│   │   └── visualization/        Plot generation
│   ├── scripts/hf_tools/         HuggingFace upload/download scripts
│   ├── experiments/              Training results & model weights
│   ├── cli.py                    CLI entrypoint
│   └── server.py                 API server entrypoint
├── frontend/
│   ├── src/
│   │   ├── features/             Explorer, classify, dataset modules
│   │   ├── pages/                Dashboard, Experiments, Classify, Dataset, Research
│   │   ├── locales/              EN + IT translations
│   │   └── components/           Layout + Shadcn/ui
│   └── package.json
├── Dockerfile
└── podman-start.sh
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify/upload` | POST | Classify uploaded image (optional `model_id`) |
| `/classify/explain` | POST | Occlusion heatmap for uploaded image |
| `/classify/models` | GET | Available trained models with metrics |
| `/api/experiments` | GET | List all experiments |
| `/api/experiments/{id}` | GET | Experiment detail with fold data |
| `/api/species` | GET | Species list with image counts |
| `/api/dataset/classes` | GET | Dataset classes with train/test counts |
| `/api/dataset/images/{split}/{class}/{file}` | GET | Serve dataset image |
| `/docs` | GET | Swagger UI |

---

## Links

- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID) · [marcello-russo](https://github.com/marcello-russo)
- **Live demo:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)
- **Dataset:** `marcellorusso/orchid-ncd-dataset`

## License

[MIT](LICENSE)
