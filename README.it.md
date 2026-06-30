[English](README.md)

<p align="center">
  <img src="assets/images/banner.png" alt="OrchID NCD Banner" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=white" alt="React"/>
  <br/>
  <em>Classificazione ultra-fine-grained e Novel Class Discovery per orchidee <b>Ophrys</b></em>
</p>

<p align="center">
  <strong>Tesi magistrale — Università degli Studi di Napoli Federico II (DIETI)</strong>
  <br/>
  <a href="https://marcellorusso-orchid-ncd.hf.space">Demo Live</a> ·
  <a href="https://github.com/squidslab/OrchID">GitHub</a>
</p>

---

## Risultati

| Modello | Exp 6 (CE) | Exp 7 (SupCon) | Exp 8 (SpHOR) | **Exp 10 AUROC** |
|---------|:---:|:---:|:---:|:---:|
| **DINOv2-Base** | 0.911 | 0.933 | 0.930 | **0.914** |
| DINOv2-Small | 0.884 | 0.913 | 0.917 | 0.815 |
| ConvNeXt-Tiny | 0.674 | 0.844 | 0.815 | 0.849 |
| ConvNeXt-Small | 0.556 | 0.704 | 0.698 | 0.846 |
| ResNet-50 | 0.887 | 0.383 | 0.540 | 0.731 |
| ResNet-18 | 0.860 | 0.363 | 0.398 | 0.680 |

AUROC Exp 10: rilevamento della classe novel (O. sphegodes\_Napoli nascosta).

---

## Avvio Rapido

### Prerequisiti
- Python 3.10+, Node.js 20+
- GPU NVIDIA (per training, inferenza CPU supportata)

### Sviluppo Locale

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

Login con `admin` / `orchid2026` (creato automaticamente al primo avvio).

### Training

```bash
cd backend
python cli.py --cross-validate --exp 6 --model resnet18
python cli.py --ncd --exp 10 --model dinov2_base
```

---

## Architettura

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Frontend   │────→│   Backend    │────→│   Model Registry │
│  React+Vite │     │  FastAPI     │     │  6 architetture  │
│  TypeScript │     │  Python 3.12 │     │  (PyTorch Hub)   │
└─────────────┘     └──────┬───────┘     └──────────────────┘
                           │
                    ┌──────┴───────┐
                    │   SQLite DB  │
                    │  (metadati)  │
                    └──────────────┘
```

**Storage:**
- **Metadati**: SQLite (`data/orchid.db`) — esperimenti, fold, utenti
- **Pesi modelli**: HuggingFace Hub (`marcellorusso/orchid-ncd-models`)
- **Dataset**: Filesystem + HuggingFace Dataset hub
- **Embedding**: File NPY per esperimento (attivazioni cached)

---

## Esperimenti

| Exp | Metodo | Fasi | Descrizione |
|-----|--------|:----:|-------------|
| 6 | CE baseline | 1 (CE) | Cross-Entropy standard, backbone + testa insieme |
| 7 | SupCon | 2 (SupCon + CE) | Apprendimento contrastivo supervisionato + linear probe |
| 8 | SpHOR | 2 (SupCon+SpHOR + CE) | Prototipi ortogonali sferici + SupCon |
| 9 | NCD (SpHOR) | 2 (idem 8) | Come 8, valutazione NCD abilitata |
| 10 | MGH-NCD | 2 (MGH-SupCon + CE) | Hard negative mining + features multi-granularità |

**Backbone**: ResNet-18/50, ConvNeXt-Tiny/Small, DINOv2-Small/Base.

**Detector OSR**: Mahalanobis (LedoitWolf), KNN (L2-normalizzato), Energy (logsumexp), Ensemble (media ranghi).

---

## Dataset

Sei specie di *Ophrys* critptiche + una classe novel:

| Specie | Train | Test | Novel |
|--------|:-----:|:----:|:----:|
| O. exaltata | ✓ | ✓ | |
| O. garganica | ✓ | ✓ | |
| O. incubacea | ✓ | ✓ | |
| O. majellensis | ✓ | ✓ | |
| O. sphegodes | ✓ | ✓ | |
| O. sphegodes\_Palena | ✓ | ✓ | |
| **O. sphegodes\_Napoli** | | ✓ | **✓** |

Download da HuggingFace:
```python
from huggingface_hub import snapshot_download
snapshot_download("marcellorusso/orchid-ncd-dataset", repo_type="dataset")
```

---

## API

| Endpoint | Metodo | Auth | Descrizione |
|----------|--------|------|-------------|
| `/auth/login` | POST | — | Login |
| `/auth/register` | POST | — | Registrazione (codice invito) |
| `/classify` | POST | — | Classifica immagine (base64) |
| `/classify/models` | GET | — | Modelli disponibili |
| `/api/experiments` | GET | — | Tutti gli esperimenti |
| `/api/experiments/{id}` | GET | — | Dettaglio esperimento con fold |
| `/api/admin/sync-db` | POST | Admin | Forza sync SQLite da JSON |
| `/api/ncd/comparison` | GET | — | Tabella risultati NCD |
| `/docs` | GET | — | Swagger UI |

---

## Link

- **Demo live:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)
- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID)
- **Modelli:** `marcellorusso/orchid-ncd-models`
- **Dataset:** `marcellorusso/orchid-ncd-dataset`

## Licenza

[MIT](LICENSE)
