<p align="center">
  <img src="assets/images/banner.png" alt="OrchID OSR Banner" width="100%"/>
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
  <em>Classificazione ultra-fine-grained e Novel Class Discovery per orchidee <b>Ophrys</b></em>
</p>

<p align="center">
  <strong>Tesi magistrale — Università degli Studi di Napoli Federico II (DIETI)</strong>
  <br/>
  <a href="https://github.com/squidslab/OrchID">GitHub</a> ·
  <a href="https://arxiv.org/abs/2503.08049">Articolo SpHOR</a>
</p>

---

## Abstract

Sei specie di *Ophrys* morfologicamente simili, una classe novel nascosta (*O. sphegodes* Napoli), 2.232 immagini di training + 300 di test. Confrontiamo 6 architetture (ResNet‑18/50, ConvNeXt‑T/S, DINOv2‑S/B) × 4 metodi (CE, SupCon, SpHOR, MGH‑NCD). **Campione closed-set:** DINOv2‑Small + SupCon (0.896 F1). **Campione open-set:** ConvNeXt‑Tiny + SpHOR (0.907 detection rate@10%FPR). Il Novel Class Discovery rimane un problema aperto (NMI 0.16–0.19).

---

## Risultati (dalla tesi)

### Exp 6 — Baseline Cross-Entropy

| Modello | F1 Test |
|---------|:-------:|
| **DINOv2‑Base** | **0.927** |
| ConvNeXt‑Tiny | 0.890 |
| DINOv2‑Small | 0.888 |
| ResNet‑50 | 0.882 |
| ResNet‑18 | 0.880 |
| ConvNeXt‑Small | 0.863 |

### Exp 7 — Supervised Contrastive Learning

| Modello | F1 Test |
|---------|:-------:|
| **DINOv2‑Small** | **0.896** |
| ResNet‑50 | 0.799 |
| ResNet‑18 | 0.736 |
| DINOv2‑Base | 0.687 |
| ConvNeXt‑Small | 0.689 |
| ConvNeXt‑Tiny | 0.681 |

### Exp 8 — SpHOR (Open-World)

| Modello | F1 Test | Detection Rate@10%FPR |
|---------|:-------:|:---------------------:|
| **ConvNeXt‑Tiny** | 0.759 | **0.907** |
| DINOv2‑Small | 0.747 | 0.670 |
| DINOv2‑Base | 0.654 | 0.746 |
| ConvNeXt‑Small | 0.668 | 0.480 |
| ResNet‑50 | 0.609 | 0.538 |
| ResNet‑18 | 0.520 | 0.093 |

### Exp 10 — MGH-NCD

| Modello | F1 Test | Detection Rate@10%FPR |
|---------|:-------:|:---------------------:|
| DINOv2‑Base | 0.369 | 0.688 |
| ConvNeXt‑Tiny | 0.349 | 0.566 |
| DINOv2‑Small | 0.270 | 0.391 |
| ConvNeXt‑Small | 0.247 | 0.536 |
| ResNet‑18 | 0.226 | 0.257 |
| ResNet‑50 | 0.174 | 0.321 |

---

## Risultati Principali

1. **Campione closed-set:** DINOv2‑Small + SupCon (0.896 F1, gap −0.01) — miglior compromesso accuratezza/generalizzazione
2. **Campione open-set:** ConvNeXt‑Tiny + SpHOR (0.907 detection rate) — prima OSR funzionante in ultra-FGVC
3. **SupCon fallisce OSR:** accuratezza classi note = **0,0** a ogni soglia (AUROC 0.485)
4. **MGH-NCD collassa:** 74% delle predizioni → *O. sphegodes* Palena; in ultra-FGVC la semplicità batte la complessità
5. **NCD irrisolto:** NMI clustering = 0.16–0.19 — risultato negativo pubblicabile

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
python cli.py --cross-validate --exp 6 --model resnet18    # CE baseline
python cli.py --cross-validate --exp 7 --model resnet18    # SupCon
python cli.py --cross-validate --exp 8 --model resnet18 --sphor   # SpHOR
python cli.py --cross-validate --exp 10 --model dinov2_base       # MGH-NCD
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
| **O. sphegodes Napoli** | — | 548 | **✓** |

Download: `snapshot_download("marcellorusso/orchid-osr-dataset", repo_type="dataset")`

---

## Deploy

- **Backend:** Docker (VPS / Dokploy) con Cloudflare Quick Tunnel
- **Frontend:** Cloudflare Pages (`orchid-osr.pages.dev`)
- **Nessun database:** tutto su filesystem (JSON, .pt, .npy)

---

## Licenza

[MIT](LICENSE)
