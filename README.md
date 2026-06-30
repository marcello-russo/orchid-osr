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

## Results

| Model | Exp 6 (CE) | Exp 7 (SupCon) | Exp 8 (SpHOR) | **Exp 10 AUROC** |
|-------|:---:|:---:|:---:|:---:|
| **DINOv2-Base** | 0.911 | 0.933 | 0.930 | **0.914** |
| DINOv2-Small | 0.884 | 0.913 | 0.917 | 0.815 |
| ConvNeXt-Tiny | 0.674 | 0.844 | 0.815 | 0.849 |
| ConvNeXt-Small | 0.556 | 0.704 | 0.698 | 0.846 |
| ResNet-50 | 0.887 | 0.383 | 0.540 | 0.731 |
| ResNet-18 | 0.860 | 0.363 | 0.398 | 0.680 |

Exp 10 AUROC: novel class detection (O. sphegodes\_Napoli held out).

---

## Quick Start

### Prerequisites
- Python 3.10+, Node.js 20+
- NVIDIA GPU (for training, CPU-only inference supported)
- [Optional] Docker + Podman

### Local Development

```bash
# Backend
cd backend
pip install -e ".[dev]"
python server.py  # → http://localhost:5000

# Frontend
cd frontend
npm install
npm run dev       # → http://localhost:5175
```

Login with `admin` / `orchid2026` (auto-created on first start).

### Training

```bash
cd backend
python cli.py --cross-validate --exp 6 --model resnet18    # CE baseline
python cli.py --cross-validate --exp 7 --model resnet18    # SupCon
python cli.py --cross-validate --exp 8 --model resnet18 --sphor  # SpHOR
python cli.py --ncd --exp 10 --model dinov2_base           # NCD evaluation
```

See [AGENTS.md](AGENTS.md) for the full command reference.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Frontend   │────→│   Backend    │────→│   Model Registry │
│  React+Vite │     │  FastAPI     │     │  (PyTorch Hub)   │
│  TypeScript │     │  Python 3.12 │     │  6 architectures │
└─────────────┘     └──────┬───────┘     └──────────────────┘
                           │
                    ┌──────┴───────┐
                    │   SQLite DB  │
                    │  Metadata    │
                    └──────────────┘
```

**Storage:**
- **Metadata**: SQLite (`data/orchid.db`) — experiments, folds, users
- **Weights**: HuggingFace Hub (`marcellorusso/orchid-ncd-models`)
- **Dataset**: Filesystem + HuggingFace Dataset hub
- **Embeddings**: NPY files per experiment (cached model activations)

**Training Pipeline:**
1. 5-fold stratified cross-validation
2. Two-phase training (SupCon + CE fine-tune) for exp 7-10
3. Early stopping (patience=15), gradient accumulation
4. NCD evaluation: embedding extraction → 4 OSR detectors → clustering → figures

---

## Experiments

| Exp | Method | Phases | Description |
|-----|--------|--------|-------------|
| 6 | CE baseline | 1 (CE) | Standard Cross-Entropy, backbone + head trained jointly |
| 7 | SupCon | 2 (SupCon + CE) | Supervised Contrastive Learning + linear probe |
| 8 | SpHOR | 2 (SupCon+SpHOR + CE) | SupCon + orthogonal spherical prototypes |
| 9 | NCD (SpHOR) | 2 (same as 8) | Same as 8, NCD evaluation enabled |
| 10 | MGH-NCD | 2 (MGH-SupCon + CE) | Hard negative mining + multi-granularity features |

**Backbones**: ResNet-18/50, ConvNeXt-Tiny/Small, DINOv2-Small/Base.

**OSR Detectors**: Mahalanobis (LedoitWolf shrinkage), KNN (L2-normalized), Energy (logsumexp), Ensemble (rank-normalized average).

---

## Dataset

Six cryptic *Ophrys* species + one held-out novel class:

| Species | Train | Test | Novel |
|---------|:-----:|:----:|:----:|
| O. exaltata | ✓ | ✓ | |
| O. garganica | ✓ | ✓ | |
| O. incubacea | ✓ | ✓ | |
| O. majellensis | ✓ | ✓ | |
| O. sphegodes | ✓ | ✓ | |
| O. sphegodes\_Palena | ✓ | ✓ | |
| **O. sphegodes\_Napoli** | | ✓ | **✓** |

Download from HuggingFace:
```python
from huggingface_hub import snapshot_download
snapshot_download("marcellorusso/orchid-ncd-dataset", repo_type="dataset")
```

---

## API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/login` | POST | — | Login |
| `/auth/register` | POST | — | Register (requires invite code) |
| `/classify` | POST | — | Classify image (base64) |
| `/classify/models` | GET | — | List available models |
| `/api/experiments` | GET | — | All experiments |
| `/api/experiments/{id}` | GET | — | Experiment detail with folds |
| `/api/comparison/experiments` | GET | — | Cross-experiment comparison |
| `/api/admin/sync-db` | POST | Admin | Force SQLite sync from JSON |
| `/api/species` | GET | — | Species with counts |
| `/api/ncd/comparison` | GET | — | NCD results table |
| `/docs` | GET | — | Swagger UI |

---

## References

- [SpHOR: Spherical Orthogonal Prototypes for Open-Set Recognition](https://arxiv.org/abs/2503.08049) — Bahavan et al., CVPR 2026 Findings
- [SupCon: Supervised Contrastive Learning](https://papers.nips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) — Khosla et al., NeurIPS 2020
- [Energy-based OOD Detection](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html) — Liu et al., NeurIPS 2020
- [Mahalanobis OOD Detection](https://arxiv.org/abs/1807.03888) — Lee et al., NeurIPS 2018
- [KNN OOD Detection](https://proceedings.mlr.press/v162/sun22d.html) — Sun et al., ICML 2022
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
