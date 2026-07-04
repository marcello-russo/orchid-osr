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
  <img src="assets/images/banner.png" alt="OrchID NCD Banner" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white" alt="SQLite"/>
  <br/>
  <em>Ultra-fine-grained classification and Novel Class Discovery for <b>Ophrys</b> orchids</em>
</p>

<p align="center">
  <strong>Master's thesis — University of Naples Federico II (DIETI)</strong>
  <br/>
  <a href="https://marcellorusso-orchid-ncd.hf.space">Live Demo</a> ·
  <a href="https://github.com/squidslab/OrchID">GitHub</a> ·
  <a href="https://arxiv.org/abs/2503.08049">SpHOR Paper</a>
</p>

---

## Abstract

Six morphologically similar *Ophrys* species, one held-out novel class (*O. sphegodes* Napoli), 2,232 training images. We benchmark 6 backbone architectures (ResNet‑18/50, ConvNeXt‑T/S, DINOv2‑S/B) × 4 methods (CE baseline, SupCon, SpHOR, MGH‑NCD) on ultra-fine-grained classification and open-set detection. **Best result: DINOv2‑Base + MGH‑NCD — 0.859 Test F1, 0.914 AUROC for novel class detection.**

---

## Results

### Classification (Test F1)

| Model | CE (Exp 6) | SupCon (Exp 7) | SpHOR (Exp 8) | MGH‑NCD (Exp 10) |
|-------|:----------:|:--------------:|:--------------:|:-----------------:|
| **DINOv2‑Base** | **0.911** | **0.933** | **0.930** | **0.859** |
| DINOv2‑Small | 0.884 | 0.913 | 0.917 | 0.850 |
| ConvNeXt‑Tiny | 0.701 | 0.844 | 0.815 | 0.602 |
| ConvNeXt‑Small | 0.527 | 0.704 | 0.698 | 0.561 |
| ResNet‑50 | 0.887 | 0.383 | 0.540 | 0.385 |
| ResNet‑18 | 0.860 | 0.363 | 0.398 | 0.386 |

### Novel Class Detection (AUROC, Ensemble detector)

| Model | SupCon (Exp 7) | SpHOR (Exp 8) | MGH‑NCD (Exp 10) |
|-------|:-------------:|:--------------:|:-----------------:|
| **DINOv2‑Base** | 0.624 | 0.912 | **0.914** |
| DINOv2‑Small | 0.584 | 0.902 | 0.815 |
| ConvNeXt‑Tiny | 0.567 | 0.792 | 0.849 |
| ConvNeXt‑Small | 0.551 | 0.785 | 0.846 |
| ResNet‑50 | 0.610 | 0.638 | 0.730 |
| ResNet‑18 | 0.553 | 0.554 | 0.680 |

Novel class: *O. sphegodes* Napoli (excluded from training).

---

## Key Findings

- **SupCon effectiveness depends on backbone pretraining alignment.** DINOv2 (self-supervised) and ConvNeXt (large-scale supervised) benefit from SupCon (+3–20% Test F1). **ResNet‑18/50 collapse under SupCon** (~0.38 vs 0.87 Test F1) because ResNet's ImageNet‑supervised features lack the contrastive alignment needed for ultra-fine-grained separation with few samples. DINOv2's self-supervised pretraining on 142M images provides embeddings that SupCon can refine effectively.
- **SpHOR improves open-set detection without sacrificing classification.** DINOv2‑Base: 0.912 AUROC (SpHOR) vs 0.624 (SupCon) — the spherical orthogonal prototypes explicitly structure the embedding space for unknown rejection.
- **MGH‑NCD matches SpHOR detection while being more robust with weaker backbones.** Hard-negative mining prevents the collapse seen in plain SupCon: ResNet‑18/50 AUROC jumps from ~0.55 to ~0.70, ConvNeXt‑T/S from ~0.57 to ~0.85.
- **MGH‑NCD trades classification accuracy for detection robustness.** The hard-negative mining distorts the embedding space, lowering DINOv2‑Base Test F1 from 0.930 (SpHOR) to 0.859. This is an expected trade-off — the model prioritises feature separability for unknown rejection over known-class compactness.
- **Best overall model for novel class detection: DINOv2‑Base + MGH‑NCD** — 0.859 Test F1 (classification), 0.914 AUROC (novel class detection).

---

## Quick Start

### Prerequisites
- Python 3.10+, Node.js 20+
- NVIDIA GPU (training only; CPU inference supported)
- [Optional] Docker

### Local Development

```bash
# Backend
cd backend
pip install -e ".[dev]"
python server.py            # → http://localhost:5000

# Frontend
cd frontend
npm install
npm run dev                 # → http://localhost:5175
```

Login with `marcello` / `orchid-2026` (auto-created on first start).

### Training

```bash
cd backend
python cli.py --cross-validate --exp 6 --model resnet18    # CE baseline
python cli.py --cross-validate --exp 7 --model resnet18    # SupCon
python cli.py --cross-validate --exp 8 --model resnet18 --sphor   # SpHOR
python cli.py --cross-validate --exp 10 --model dinov2_base        # MGH-NCD
python cli.py --ncd --exp 10 --model dinov2_base                  # NCD evaluation
```

See [AGENTS.md](AGENTS.md) for the full command reference.

---

## Experiments

| Exp | Method | Phases | Description |
|-----|--------|--------|-------------|
| 6 | CE baseline | 1 (CE) | Standard cross-entropy, backbone + head trained jointly |
| 7 | SupCon | 2 (SupCon → CE) | Supervised contrastive pretraining → linear probe |
| 8 | SpHOR | 2 (SupCon+SpHOR → CE) | Orthogonal spherical prototypes for open-set regularisation |
| 10 | MGH‑NCD | 2 (MGH‑SupCon → CE) | Hard-negative mining + multi-granularity features |

All experiments use 5-fold stratified cross-validation with seed 57.

**Backbones (6):** ResNet‑18/50, ConvNeXt‑Tiny/Small, DINOv2‑Small/Base.

---

## Dataset

Six cryptic *Ophrys* species + one held-out novel class:

| Species | Train | Test | Novel |
|---------|:-----:|:----:|:-----:|
| O. exaltata | ✓ | ✓ | |
| O. garganica | ✓ | ✓ | |
| O. incubacea | ✓ | ✓ | |
| O. majellensis | ✓ | ✓ | |
| O. sphegodes | ✓ | ✓ | |
| O. sphegodes Palena | ✓ | ✓ | |
| **O. sphegodes Napoli** | | ✓ | **✓** |

Download from HuggingFace:
```python
from huggingface_hub import snapshot_download
snapshot_download("marcellorusso/orchid-ncd-dataset", repo_type="dataset")
```

---

## Architecture

```
Frontend (React+Vite) ────→ Backend (FastAPI) ────→ Model Registry (PyTorch)
                                │
                                ↓
                           SQLite DB
                        (Metadata, users)
```

**Storage breakdown:**
- **Metadata**: SQLite (`data/orchid.db`) — experiments, folds, users
- **Weights**: HuggingFace Hub (`marcellorusso/orchid-ncd-models`) — 144 `.pt` files (5 folds × 6 models × 4 exps + adapters)
- **Dataset**: Filesystem + HuggingFace Datasets
- **Embeddings**: NPY files per experiment (cached activations for NCD)

---

## API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/login` | POST | — | Login |
| `/auth/register` | POST | — | Register (requires invite code from admin) |
| `/classify` | POST | — | Classify image (base64) |
| `/classify/models` | GET | — | Available models |
| `/api/experiments` | GET | — | List all experiments |
| `/api/experiments/{id}` | GET | — | Experiment detail with folds |
| `/api/experiments/{id}/fold/{n}` | GET | — | Single fold data |
| `/api/comparison/experiments` | GET | — | Cross-experiment comparison |
| `/api/ncd/comparison` | GET | — | NCD results table |
| `/api/embeddings` | GET | — | Latent space viewer data |
| `/api/species` | GET | — | Species with image counts |
| `/api/admin/sync-db` | POST | Admin | Force SQLite sync from JSON |
| `/docs` | GET | — | Swagger UI |

---

## References

- [SpHOR: Spherical Orthogonal Prototypes for Open-Set Recognition](https://arxiv.org/abs/2503.08049) — Bahavan et al., CVPR 2026 Findings
- [SupCon: Supervised Contrastive Learning](https://papers.nips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) — Khosla et al., NeurIPS 2020
- [MGCE: Generalized Fine-Grained Category Discovery](https://arxiv.org/abs/2509.26227) — Zheng et al., 2025
- [Generalized Category Discovery](https://doi.org/10.1109/CVPR52688.2022.00737) — Vaze et al., CVPR 2022

---

## Links

- **Live demo:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)
- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID)
- **Models:** `marcellorusso/orchid-ncd-models`
- **Dataset:** `marcellorusso/orchid-ncd-dataset`

## License

[MIT](LICENSE)
