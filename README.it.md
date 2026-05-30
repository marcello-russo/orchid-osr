[English](README.md)

<p align="center">
  <img src="assets/images/banner.png" alt="OrchID NCD Banner" width="100%"/>
</p>

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
  <img src="https://img.shields.io/badge/HuggingFace_Spaces-FFD21E?style=flat&logo=huggingface&logoColor=000" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white" alt="GitHub Actions"/>
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA"/>
</p>

# OrchID-NCD

> Classificazione visiva ultra-fine-grained e Novel Class Discovery per orchidee *Ophrys*.

Progetto di tesi magistrale — Università degli Studi di Napoli Federico II (DIETI).

**Demo live:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)

---

## Funzionalità

- **6 architetture deep learning** — ResNet, ConvNeXt, DINOv2 con ottimizzatori personalizzati e LoRA
- **Multi-esperimento** — CE baseline, SupCon, prototipi sferici ortogonali, MGH-NCD
- **Novel Class Discovery** — Open-Set Recognition (Mahalanobis + KNN + Energy), clustering, t-SNE/UMAP
- **Feature multi-granularità** — attivazioni concatenate da più stadi della rete
- **Hard negative mining** — peso maggiore sulle coppie di classi visivamente simili
- **Autenticazione su invito** — admin genera token, utenti si registrano con username/password, bcrypt
- **Internazionalizzazione** — italiano e inglese
- **Dashboard interattiva** — KPI esperimenti, distribuzione specie, metriche per-fold, risultati NCD
- **Galleria dataset** — esplora le 6 specie con split train/test
- **Classificazione immagini** — upload o drag-and-drop, selezione modello, cache pesi
- **Explainability** — mappe di calore Captum
- **CI/CD** — GitHub Actions con ruff, pytest, ESLint, Docker build

---

## Quick Start

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/orchid-ncd

# Backend
cd backend && python server.py --reload

# Frontend
cd frontend && npm install && npm run dev
```

---

## API

| Endpoint | Metodo | Auth | Descrizione |
|----------|--------|------|-------------|
| `/auth/login` | POST | — | Login |
| `/auth/register` | POST | — | Registrazione con codice invito |
| `/classify` | POST | Pubblico | Classifica immagine (base64) |
| `/classify/upload` | POST | Pubblico | Classifica file caricato |
| `/classify/explain` | POST | User | Mappa di calore Captum |
| `/classify/models` | GET | Pubblico | Modelli disponibili |
| `/api/experiments` | GET | Pubblico | Tutti gli esperimenti |
| `/api/species` | GET | Pubblico | Specie con conteggio |
| `/api/comparison` | GET | Pubblico | Confronto cross-experiment |
| `/api/dataset/classes` | GET | User | Conteggio classi |
| `/api/embeddings` | GET | User | Spazio latente (t-SNE/UMAP) |
| `/api/ncd/{model}/results` | GET | User | Risultati NCD |
| `/auth/me` | GET | Bearer | Utente corrente |
| `/docs` | GET | — | Swagger UI |

---

## Licenza

[MIT](LICENSE)
