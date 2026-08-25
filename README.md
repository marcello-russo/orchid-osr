<p align="center">
  <img src="assets/images/banner.webp" alt="OrchID OSR Banner" width="100%"/>
</p>

<p align="center">
  <a href="README.it.md">🇮🇹 Italiano</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white" alt="SQLite"/>
  <br/>
  <em>Ultra-fine-grained classification and open-set recognition for <b>Ophrys</b> orchids via deep contrastive learning</em>
</p>

<p align="center">
  <strong>Master's thesis — University of Naples Federico II (DIETI)</strong>
  <br/>
  <a href="https://github.com/squidslab/OrchID">GitHub</a> ·
  <a href="https://arxiv.org/abs/2503.08049">SpHOR Paper</a>
</p>

---

## Abstract

Six morphologically similar *Ophrys* species, one held-out novel class (*O. garganica*), 2,232 training + 300 test images. We benchmark 6 architectures (ResNet‑18/50, ConvNeXt‑T/S, DINOv2‑S/B) × 3 learning paradigms (CE, SupCon, SpHOR). **Closed-set champion:** DINOv2‑Base + SupCon (0.936 F1 test). **Open-set champion:** DINOv2‑Base + SpHOR (AUROC 0.875 on the held-out novel class). Open-set recognition remains architecture-dependent and far from solved in this ultra-fine-grained domain.

---

## Results (from thesis)

### Exp 6 — Cross-Entropy Baseline

| Model | F1 Test |
|-------|:-------:|
| **ResNet‑50** | **0.980** |
| ResNet‑18 | 0.978 |
| DINOv2‑Base | 0.975 |
| DINOv2‑Small | 0.970 |
| ConvNeXt‑Tiny | 0.821 |
| ConvNeXt‑Small | 0.793 |

### Exp 7 — Supervised Contrastive Learning

| Model | F1 Test |
|-------|:-------:|
| **DINOv2‑Base** | **0.936** |
| DINOv2‑Small | 0.888 |
| ConvNeXt‑Tiny | 0.837 |
| ConvNeXt‑Small | 0.517 |
| ResNet‑50 | 0.465 |
| ResNet‑18 | 0.444 |

### Exp 8 — SpHOR (Open-World)

| Model | F1 Test |
|-------|:-------:|
| **ConvNeXt‑Tiny** | **0.911** |
| DINOv2‑Base | 0.910 |
| DINOv2‑Small | 0.894 |
| ConvNeXt‑Small | 0.874 |
| ResNet‑50 | 0.750 |
| ResNet‑18 | 0.560 |

### Exp 9 — NCD (Open-Set Recognition)

| Model | F1 Test | AUROC (best detector) |
|-------|:-------:|:---------------------:|
| **DINOv2‑Base** | **0.932** | **0.875** (Ensemble) |
| ConvNeXt‑Small | 0.933 | 0.840 (Ensemble) |
| DINOv2‑Small | 0.921 | 0.760 (KNN) |
| ConvNeXt‑Tiny | 0.920 | 0.850 (Ensemble) |
| ResNet‑50 | 0.856 | 0.776 (KNN) |
| ResNet‑18 | 0.590 | 0.507 (Mahalanobis) |

*Exp 9 excludes O. garganica from training, leaving it as the hidden novel class for open-set evaluation.*

---

## Key Findings

1. **Closed-set champion:** DINOv2‑Base + SupCon (0.936 F1 test) — best accuracy among contrastive paradigms
2. **Open-set champion:** DINOv2‑Base + SpHOR (AUROC 0.875 on the held-out novel class) — best OSR in ultra-FGVC
3. **OSR is architecture-dependent:** best AUROC ranges from 0.507 (ResNet‑18) to 0.875 (DINOv2‑Base); SpHOR does not consistently beat chance
4. **SupCon OSR failure:** the SupCon detector alone fails on known classes (AUROC 0.485) — spherical prototypes are required for open-set regularisation
5. **NCD partially solved:** clustering recovers species when the metric space is stable (NMI up to 0.84) — a promising but not fully solved result

---

## Quick Start

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

### Training

```bash
cd backend
python cli.py --cross-validate --exp 6 --model resnet18    # CE baseline
python cli.py --cross-validate --exp 7 --model resnet18    # SupCon
python cli.py --cross-validate --exp 8 --model resnet18 --sphor   # SpHOR
python cli.py --cross-validate --exp 9 --model dinov2_small --sphor   # NCD (hidden novel class)
python cli.py --ncd --exp 9 --model dinov2_small                  # OSR + clustering evaluation
```

See [AGENTS.md](AGENTS.md) for the full command reference.

---

## Experiments

| Exp | Method | Phases | Description |
|-----|--------|--------|-------------|
| 6 | CE baseline | 1 (CE) | Standard cross-entropy, backbone + head trained jointly |
| 7 | SupCon | 2 (SupCon → CE) | Supervised contrastive pretraining → linear probe |
| 8 | SpHOR | 2 (SupCon+SpHOR → CE) | Orthogonal spherical prototypes for open-set regularisation |
| 9 | NCD | 2 (SupCon+SpHOR → CE) | O. garganica excluded as hidden novel class; OSR + clustering evaluation |

All experiments use 5-fold stratified CV with seed 57 over 6 backbones.

---

## Dataset

| Species | Train | Test | Novel |
|---------|:-----:|:----:|:-----:|
| O. exaltata | 501 | 50 | |
| O. garganica | 570 | 50 | |
| O. incubacea | 366 | 50 | |
| O. majellensis | 152 | 50 | |
| O. sphegodes | 433 | 50 | |
| O. sphegodes Palena | 210 | 50 | |

*O. garganica (row above) is used as the hidden novel class in Exp 9: excluded from training, reserved for test only.*

Download from HuggingFace: `snapshot_download("marcellorusso/orchid-osr-dataset", repo_type="dataset")`

---

## Deploy

- **Backend:** Docker (VPS / Dokploy) behind Cloudflare Quick Tunnel
- **Frontend:** Cloudflare Pages (`orchid-osr.pages.dev`)
- **No database:** filesystem-based (experiments JSON, model .pt files, embeddings .npy)

---

## References

- [SpHOR: Spherical Orthogonal Prototypes for Open-Set Recognition](https://arxiv.org/abs/2503.08049) — CVPR 2026 Findings
- [SupCon: Supervised Contrastive Learning](https://papers.nips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) — NeurIPS 2020
- [MGCE: Generalized Fine-Grained Category Discovery](https://arxiv.org/abs/2509.26227) — 2025
- [Generalized Category Discovery](https://openaccess.thecvf.com/content/CVPR2022/html/Vaze_Generalized_Category_Discovery_CVPR_2022_paper.html) — CVPR 2022

---

## Links

- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID)
- **Models:** `marcellorusso/orchid-osr-models`
- **Dataset:** `marcellorusso/orchid-osr-dataset`

## License

[MIT](LICENSE)
