<p align="center">
  <img src="assets/images/banner.png" alt="OrchidOSR NCD Banner" width="100%"/>
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
  <em>Ultra-fine-grained classification and Novel Class Discovery for <b>Ophrys</b> orchids</em>
</p>

<p align="center">
  <strong>Master's thesis — University of Naples Federico II (DIETI)</strong>
  <br/>
  <a href="https://github.com/squidslab/OrchidOSR">GitHub</a> ·
  <a href="https://arxiv.org/abs/2503.08049">SpHOR Paper</a>
</p>

---

## Abstract

Six morphologically similar *Ophrys* species, one held-out novel class (*O. sphegodes* Napoli), 2,232 training + 300 test images. We benchmark 6 architectures (ResNet‑18/50, ConvNeXt‑T/S, DINOv2‑S/B) × 4 methods (CE, SupCon, SpHOR, MGH‑NCD). **Closed-set champion:** DINOv2‑Small + SupCon (0.896 F1). **Open-set champion:** ConvNeXt‑Tiny + SpHOR (0.907 detection rate@10%FPR). Novel Class Discovery remains unsolved (NMI 0.16–0.19).

---

## Results (from thesis)

### Exp 6 — Cross-Entropy Baseline

| Model | F1 Test |
|-------|:-------:|
| **DINOv2‑Base** | **0.927** |
| ConvNeXt‑Tiny | 0.890 |
| DINOv2‑Small | 0.888 |
| ResNet‑50 | 0.882 |
| ResNet‑18 | 0.880 |
| ConvNeXt‑Small | 0.863 |

### Exp 7 — Supervised Contrastive Learning

| Model | F1 Test |
|-------|:-------:|
| **DINOv2‑Small** | **0.896** |
| ResNet‑50 | 0.799 |
| ResNet‑18 | 0.736 |
| DINOv2‑Base | 0.687 |
| ConvNeXt‑Small | 0.689 |
| ConvNeXt‑Tiny | 0.681 |

### Exp 8 — SpHOR (Open-World)

| Model | F1 Test | Detection Rate@10%FPR |
|-------|:-------:|:---------------------:|
| **ConvNeXt‑Tiny** | 0.759 | **0.907** |
| DINOv2‑Small | 0.747 | 0.670 |
| DINOv2‑Base | 0.654 | 0.746 |
| ConvNeXt‑Small | 0.668 | 0.480 |
| ResNet‑50 | 0.609 | 0.538 |
| ResNet‑18 | 0.520 | 0.093 |

### Exp 10 — MGH-NCD

| Model | F1 Test | Detection Rate@10%FPR |
|-------|:-------:|:---------------------:|
| DINOv2‑Base | 0.369 | 0.688 |
| ConvNeXt‑Tiny | 0.349 | 0.566 |
| DINOv2‑Small | 0.270 | 0.391 |
| ConvNeXt‑Small | 0.247 | 0.536 |
| ResNet‑18 | 0.226 | 0.257 |
| ResNet‑50 | 0.174 | 0.321 |

---

## Key Findings

1. **Closed-set champion:** DINOv2‑Small + SupCon (0.896 F1, gap −0.01) — best accuracy/generalization trade-off
2. **Open-set champion:** ConvNeXt‑Tiny + SpHOR  (0.907 detection rate) — first working OSR in ultra-FGVC
3. **SupCon OSR failure:** accuracy on known classes = **0.0** at any threshold (AUROC 0.485)
4. **MGH-NCD collapse:** 74% of predictions → *O. sphegodes* Palena; simplicity beats complexity in ultra-FGVC
5. **NCD unsolved:** clustering NMI = 0.16–0.19 — a publishable negative result

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
python cli.py --cross-validate --exp 10 --model dinov2_base       # MGH-NCD
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
| **O. sphegodes Napoli** | — | 548 | **✓** |

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

- **GitHub:** [squidslab/OrchidOSR](https://github.com/squidslab/OrchidOSR)
- **Models:** `marcellorusso/orchid-osr-models`
- **Dataset:** `marcellorusso/orchid-osr-dataset`

## License

[MIT](LICENSE)
