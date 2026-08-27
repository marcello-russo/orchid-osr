<p align="center">
  <img src="assets/images/banner.webp" alt="OrchID OSR Banner" width="100%"/>
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white" alt="SQLite"/>
  <br/>
  <em>Classificazione ultra-fine-grained e riconoscimento open-set per orchidee <b>Ophrys</b> tramite apprendimento contrastivo profondo</em>
</p>

<p align="center">
  <strong>Tesi magistrale — Università degli Studi di Napoli Federico II (DIETI)</strong>
  <br/>
  <a href="https://github.com/squidslab/OrchID">GitHub</a> ·
  <a href="https://arxiv.org/abs/2503.08049">Articolo SpHOR</a>
</p>

---

## Abstract

Sei specie di *Ophrys* morfologicamente simili, una classe novel nascosta (*O. garganica*), 2.192 immagini di training + 300 di test. Confrontiamo 6 architetture (ResNet‑18/50, ConvNeXt‑T/S, DINOv2‑S/B) × 3 paradigmi di apprendimento (CE, SupCon, SpHOR). **Campione closed-set:** DINOv2‑Base + SupCon (0.936 F1 test). **Campione open-set:** DINOv2‑Base + SpHOR (AUROC 0.875 sulla novel class nascosta). L'open-set recognition rimane dipendente dall'architettura e lontano dall'essere risolto in questo dominio ultra-fine-grained.

---

## Risultati (dalla tesi)

### Exp 1 — Baseline Cross-Entropy

| Modello | F1 Test |
|---------|:-------:|
| **ResNet‑50** | **0.980** |
| ResNet‑18 | 0.978 |
| DINOv2‑Base | 0.975 |
| DINOv2‑Small | 0.970 |
| ConvNeXt‑Tiny | 0.821 |
| ConvNeXt‑Small | 0.793 |

### Exp 2 — Supervised Contrastive Learning

| Modello | F1 Test |
|---------|:-------:|
| **DINOv2‑Base** | **0.936** |
| DINOv2‑Small | 0.888 |
| ConvNeXt‑Tiny | 0.837 |
| ConvNeXt‑Small | 0.517 |
| ResNet‑50 | 0.465 |
| ResNet‑18 | 0.444 |

### Exp 3 — SpHOR (Open-World)

| Modello | F1 Test |
|---------|:-------:|
| **ConvNeXt‑Tiny** | **0.911** |
| DINOv2‑Base | 0.910 |
| DINOv2‑Small | 0.894 |
| ConvNeXt‑Small | 0.874 |
| ResNet‑50 | 0.750 |
| ResNet‑18 | 0.560 |

### Exp 4 — NCD (Open-Set Recognition)

| Modello | F1 Test | AUROC (miglior rivelatore) |
|---------|:-------:|:--------------------------:|
| **DINOv2‑Base** | **0.932** | **0.875** (Ensemble) |
| ConvNeXt‑Small | 0.933 | 0.840 (Ensemble) |
| DINOv2‑Small | 0.921 | 0.760 (KNN) |
| ConvNeXt‑Tiny | 0.920 | 0.850 (Ensemble) |
| ResNet‑50 | 0.856 | 0.776 (KNN) |
| ResNet‑18 | 0.590 | 0.507 (Mahalanobis) |

*Exp 4 esclude O. garganica dal training, lasciandola come novel class nascosta per la valutazione open-set.*

---

## Risultati Principali

1. **Campione closed-set:** DINOv2‑Base + SupCon (0.936 F1 test) — migliore accuratezza tra i paradigmi contrastivi
2. **Campione open-set:** DINOv2‑Base + SpHOR (AUROC 0.875 sulla novel class nascosta) — migliore OSR in ultra-FGVC
3. **OSR dipendente dall'architettura:** l'AUROC varia da 0.507 (ResNet‑18) a 0.875 (DINOv2‑Base); SpHOR non supera sistematicamente il caso
4. **SupCon fallisce OSR:** il rivelatore SupCon da solo fallisce sulle classi note (AUROC 0.485) — i prototipi sferici sono necessari per la regolarizzazione open-set
5. **NCD parzialmente risolto:** il clustering recupera le specie quando lo spazio metrico è stabile (NMI fino a 0.84) — un risultato promettente ma non del tutto risolto

---

## Avvio Rapido

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
python cli.py --cross-validate --exp 1 --model resnet18    # CE baseline
python cli.py --cross-validate --exp 2 --model resnet18    # SupCon
python cli.py --cross-validate --exp 3 --model resnet18 --sphor   # SpHOR
python cli.py --cross-validate --exp 4 --model dinov2_small --sphor   # NCD (novel class nascosta)
python cli.py --ncd --exp 4 --model dinov2_small                  # valutazione OSR + clustering
```

Vedi [AGENTS.md](AGENTS.md) per tutti i comandi.

---

## Dataset

| Specie | Train | Test | Novel |
|--------|:-----:|:----:|:-----:|
| O. exaltata | 501 | 50 | |
| O. garganica | 570 | 50 | |
| O. incubacea | 366 | 50 | |
| O. majellensis | 152 | 50 | |
| O. sphegodes | 433 | 50 | |
| O. sphegodes Palena | 210 | 50 | |

*O. garganica (riga precedente) è usata come novel class nascosta in Exp 4: esclusa dal training, riservata solo al test.*

Download: `snapshot_download("marcellorusso/orchid-osr-dataset", repo_type="dataset")`

---

## Deploy

- **Backend:** Docker (VPS / Dokploy) con Cloudflare Quick Tunnel
- **Frontend:** Cloudflare Pages (`orchid-osr.pages.dev`)
- **Nessun database:** tutto su filesystem (JSON, .pt, .npy)

---

## Licenza

[MIT](LICENSE)
