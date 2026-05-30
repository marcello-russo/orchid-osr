[English](README.md)

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

> Classificazione visiva ultra-fine-grained e representation learning delle specie di orchidee *Ophrys* tramite self-supervised learning e tecniche contrastive.

Progetto di tesi magistrale presso l'Università degli Studi di Napoli Federico II (DIETI).

**Demo live:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)

---

## Il problema Ophrys

Il genere *Ophrys* è uno dei gruppi di orchidee più tassonomicamente complessi del Mediterraneo. Usano la decezione sessuale per attirare impollinatori specie-specifici — i fiori imitano le femmine degli insetti attraverso segnali visivi, olfattivi e tattili. Questa estrema specializzazione porta a divergenza morfologica rapida e ibridazioni frequenti, producendo esemplari visivamente ambigui che sfidano anche botanici esperti.

Il complesso *Ophrys sphegodes* da solo comprende taxa crittici multipli con tratti fenotipici sovrapposti, rendendolo un banco di prova ideale per la classificazione ultra-fine-grained e la scoperta di classi nascoste (Novel Class Discovery).

---

## Specie

Sei taxa del complesso *Ophrys sphegodes*:

| Taxon | Immagini | Fonte |
|-------|----------|-------|
| *O. exaltata* | 501 train / 50 test | privato + internet |
| *O. garganica* | 570 train / 50 test | privato + internet |
| *O. incubacea* | 366 train / 50 test | privato + internet |
| *O. majellensis* | 152 train / 50 test | solo privato |
| *O. sphegodes* | 433 train / 50 test | privato + internet |
| *O. sphegodes* (Palena) | 210 train / 50 test | solo privato |
| **Totale** | **2232 / 300** | |

---

## Pipeline

Pipeline di preprocessing modulare — ogni step è indipendente e riprendibile:

```
Immagini raw (privato + internet)
        |
        v
   [1] CROP          --crop
   Faster R-CNN per rilevamento fiori
        |
        v
   [2] SPLIT         --split
   Split pulito + dedup MD5
        |
        v
   [3] RESIZE        --resize
   Letterbox per architettura
        |
        v
   [4] TRAIN         --cross-validate --exp 10
   Stratified 5-fold CV
```

**Decisioni chiave:**
- **Cropping fiori:** Faster R-CNN (ResNet50-FPN) rileva e croppa i fiori dalle foto full-resolution
- **Zero data leakage:** hash MD5 garantisce nessuna sovrapposizione tra train e test
- **Resize letterbox:** dimensioni specifiche per architettura (ResNet 256×512, ConvNeXt/DINOv2 224×224)

---

## Esperimenti

| Exp | Metodo | Descrizione | Stato |
|-----|--------|-------------|-------|
| 6 | CE Baseline | Cross-entropy su clean split, 6 architetture | ✅ Completato |
| 7 | SupCon | Supervised Contrastive Learning + CE head | ✅ Completato |
| 8 | SpHOR | SupCon + prototipi sferici ortogonali | ✅ Completato |
| 9 | NCD | Exp 8 senza O. majellensis (classe nascosta) | ✅ Completato |
| 10 | **MGH-NCD** | Multi-Granularity Hard-negative NCD | ✅ Completato |

### Exp 10 — MGH-NCD

Multi-Granularity Hard-negative Novel Class Discovery:
- **Feature multi-granularità:** concatenazione di output da stadi multipli della rete (es. stage3 + stage4)
- **Hard negative mining:** peso 5× sulle coppie (majellensis, sphegodes) e 3× su (sphegodes, Palena)
- **Full fine-tune:** backbone scongelato in Phase 2 per modelli non ViT
- **LoRA r=32:** applicato in Phase 1 per adattamento efficiente dei backbone ViT
- **OSR Ensemble:** Mahalanobis + KNN + Energy per rilevamento open-set

---

## Architetture

| Famiglia | Modelli | Motivazione |
|----------|---------|-------------|
| **ResNet** | ResNet18, ResNet50 | Feature gerarchiche locali tramite connessioni residuali |
| **ConvNeXt** | ConvNeXt-Tiny, ConvNeXt-Small | CNN modernizzata con kernel grandi e LayerNorm |
| **DINOv2** | DINOv2-Small, DINOv2-Base | ViT self-supervisionato con attenzione globale, fine-tune via LoRA |

<details>
<summary>Ricette di training per architettura</summary>

| Modello | Optimizer | LR | Weight Decay |
|---------|-----------|----|-------------|
| ResNet18/50 | Adam | 0.001 | 0 |
| ConvNeXt-Tiny/Small | AdamW | 0.0003 | 0.05 |
| DINOv2-Small/Base | AdamW | 0.0002 | 0.01 |

Condiviso: 5-fold CV stratificato, augmentation bilanciata per fold, patience=15.

</details>

---

## Risultati

### Exp 10 — MGH-NCD (Test Set)

| Modello | Accuratezza | F1 Macro | Best Fold |
|---------|-------------|----------|-----------|
| ResNet18 | — | — | fold 4 |
| ResNet50 | 0.35 | 0.32 | fold 3 |
| ConvNeXt-Tiny | — | — | fold 2 |
| ConvNeXt-Small | — | — | fold 3 |
| DINOv2-Small | — | — | fold 4 |
| DINOv2-Base | 0.73 | 0.70 | fold 1 |

### Rilevamento O. majellensis

| Modello | Recall | F1 |
|---------|--------|----|
| DINOv2-Base | 0.16 | 0.28 |
| ResNet50 | 0.00 | 0.00 |

*Majellensis rimane la classe più difficile — visivamente quasi identica a sphegodes.*

---

## Autenticazione

L'app supporta registrazione su invito con tre ruoli:

| Ruolo | Accesso |
|-------|---------|
| **Pubblico** | Health, classify, esperimenti, specie, comparison, figure NCD |
| **Registrato** | Tutto pubblico + classi dataset, explain, embeddings, risultati NCD |
| **Admin** | Tutto + immagini dataset, gestione cache |

### Setup

```bash
cd backend

# Genera token invito
python cli.py --gen-invites 5

# Crea utente direttamente
python cli.py --add-user username --password pass --role user

# Lista utenti e inviti
python cli.py --list-users
```

Le password sono hashate con bcrypt e salvate in `data/users.json` (gitignorato). Su HF Spaces il file è persistito in `/data/` e sincronizzato col dataset privato.

---

## Dataset

Disponibile su HuggingFace: `marcellorusso/orchid-ncd-dataset` (privato)

```python
from huggingface_hub import snapshot_download
snapshot_download("marcellorusso/orchid-ncd-dataset", repo_type="dataset", local_dir="dataset")
```

---

## Quick Start

### Prerequisiti
- Linux (testato su Ubuntu/Pop!_OS)
- Docker (o Docker) + NVIDIA GPU + Container Toolkit

### Esecuzione

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/orchid-ncd
./docker-start.sh
```

### Sviluppo

```bash
# Terminale 1 — Backend
cd backend
python server.py --reload

# Terminale 2 — Frontend (Vite dev server su :5175)
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

# Smart trainer (tutti i modelli)
./smart-train.sh --exp 10
```

---

## Tech Stack

| Layer | Tecnologia |
|-------|-----------|
| Training | PyTorch + CUDA, torchvision, Captum |
| Modelli | ResNet18/50, ConvNeXt-Tiny/Small, DINOv2-Small/Base |
| Backend | FastAPI, Python 3.12 |
| Frontend | React 19, TypeScript, Vite, Tailwind v4, Shadcn/ui, Recharts |
| Infra | Docker + NVIDIA Container Toolkit, HuggingFace Spaces |

---

## API

| Endpoint | Metodo | Auth | Descrizione |
|----------|--------|------|-------------|
| `/auth/register` | POST | — | Registrazione con codice invito |
| `/auth/login` | POST | — | Login, restituisce JWT |
| `/auth/me` | GET | Bearer | Info utente corrente |
| `/classify/upload` | POST | Pubblico | Classifica immagine caricata |
| `/classify/explain` | POST | User | Mappa di calore (Captum) |
| `/classify/models` | GET | Pubblico | Modelli disponibili |
| `/api/experiments` | GET | Pubblico | Lista esperimenti |
| `/api/species` | GET | Pubblico | Specie con conteggio immagini |
| `/api/dataset/classes` | GET | User | Conteggio classi dataset |
| `/api/dataset/images/{split}/{class}/{file}` | GET | Admin | Immagine dataset |
| `/docs` | GET | — | Swagger UI |

---

## Link

- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID)
- **Demo live:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)
- **Modelli:** `marcellorusso/orchid-ncd-models`
- **Dataset:** `marcellorusso/orchid-ncd-dataset`

## Licenza

[MIT](LICENSE)
