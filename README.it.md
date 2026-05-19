[English](README.md)

# OrchID-NCD — Orchid Representation Learning & Novel Class Discovery

> Classificazione visiva ultra-fine-grained e representation learning delle specie di orchidee *Ophrys* tramite self-supervised learning e tecniche contrastive.

Progetto di Tesi Magistrale presso l'Università degli Studi di Napoli Federico II (Dipartimento DIETI).

**Demo live:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)

---

## Il Problema Ophrys

Il genere *Ophrys* rappresenta uno dei gruppi di orchidee più complessi dal punto di vista tassonomico nel Mediterraneo. Queste orchidee utilizzano l'inganno sessuale per attirare impollinatori specie-specifici — i fiori mimano insetti femmine attraverso segnali visivi, olfattivi e tattili. Questa estrema specializzazione guida una rapida divergenza morfologica e frequenti ibridazioni, producendo esemplari visivamente ambigui che sfidano anche i botanici esperti.

Il solo complesso *Ophrys sphegodes* comprende taxa crittici multipli con tratti fenotipici sovrapposti, rendendolo un banco di prova ideale per la classificazione ultra-fine-grained e la novel class discovery.

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

La pipeline di preprocessing è una CLI modulare — ogni step è indipendente e riprendibile:

```
Immagini raw (private + internet)
        |
        v
   [1] CROP          --crop
   Rilevamento fiore Faster R-CNN
        |
        v
   [2] SPLIT         --split
   Clean split + dedup MD5
        |
        v
   [3] RESIZE        --resize
   Letterbox per-architettura
        |
        v
   [4] TRAIN         --cross-validate --exp 6
   5-fold stratified CV
```

**Scelte progettuali:**

- **Cropping del fiore:** Faster R-CNN (ResNet50-FPN) rileva e ritaglia i fiori dalle foto a risoluzione piena, focalizzando il modello sulle strutture floreali discriminanti
- **Zero data leakage:** L'hashing MD5 del contenuto garantisce nessuna sovrapposizione tra train e test — gli esperimenti precedenti (exp 1–5) sono stati invalidati quando questo è stato scoperto
- **Letterbox resizing:** Le immagini sono scalate per rientrare nelle dimensioni per-architettura preservando il rapporto d'aspetto, con padding nero a riempire lo spazio rimanente
- **Dimensioni per-architettura:** ResNet usa 256x512, ConvNeXt e DINOv2 usano 224x224 — le dimensioni derivano dal registry, non sono hardcoded

---

## Roadmap

| Fase | Obiettivo | Stato |
|------|-----------|-------|
| **F1 — Classificazione** | 6 architetture, k-fold CV, recipe per-architettura, pipeline cropper Faster R-CNN | Completata |
| **F1b — Explorer** | Dashboard React SPA, classificazione upload, explainability, galleria dataset, i18n IT/EN | Completata |
| **F2 — Spazio Vettoriale** | Supervised Contrastive Learning (SupCon) per strutturare lo spazio latente | Prossima |
| **F3 — Novel Class Discovery** | Ricerca nearest-neighbor, clustering specie, recognition open-set | Pianificata |

---

## Comparazione Architetture

Tre famiglie di architetture con bias induttivi diversi:

| Famiglia | Modelli | Perché |
|----------|---------|--------|
| **ResNet** | ResNet18, ResNet50 | Feature locali gerarchiche tramite connessioni residue — baseline di fine-tuning collaudata |
| **ConvNeXt** | ConvNeXt-Tiny, ConvNeXt-Small | CNN modernizzata con kernel ampi e LayerNorm — bias induttivo più forte per la texture |
| **DINOv2** | DINOv2-Small, DINOv2-Base | ViT self-supervisionato con attenzione globale, addestrato su 142M immagini — fine-tunato via adapter LoRA (~0.5% parametri trainable) |

<details>
<summary>Recipe di training per-architettura</summary>

| Modello | Ottimizzatore | LR | Weight Decay | Motivazione |
|---------|--------------|----|-------------|-------------|
| ResNet18/50 | Adam | 0.001 | 0 | BatchNorm fornisce regolarizzazione implicita |
| ConvNeXt-Tiny/Small | AdamW | 0.0001 | 0.05 | Recipe del paper ConvNeXt — LayerNorm sensibile a gradienti grandi |
| DINOv2-Small/Base | AdamW | 0.0002 | 0.01 | Leggera regolarizzazione su adapter LoRA |

Comuni: batch fisico=8 (effettivo 32 via gradient accumulation), patience=15 su val_loss, 5-fold stratified CV, augmentation bilanciata per-fold.

</details>

---

## Dataset

Disponibile su HuggingFace: `marcellorusso/orchid-ncd-dataset` (privato)

```
raw/
  private/                    Foto di campo originali (6 classi, 1252 immagini)
  internet/                   Immagini da internet (4 classi, 1080 immagini)
cropped/
  256x512/                    Letterbox per ResNet
    train/  test/
  224x224/                    Letterbox per ConvNeXt / DINOv2
    train/  test/
```

```python
from huggingface_hub import snapshot_download
snapshot_download("marcellorusso/orchid-ncd-dataset", repo_type="dataset", local_dir="dataset")
```

---

## Avvio Rapido

### Prerequisiti
- Linux (testato su Ubuntu/Pop!_OS)
- Podman (o Docker) + NVIDIA GPU + Container Toolkit

### Esecuzione

```bash
git clone https://github.com/squidslab/OrchID.git
cd OrchID/resnet18
./podman-start.sh
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

# Pipeline di preprocessing (step singoli)
python cli.py --crop                    # Cropping fiore con Faster R-CNN
python cli.py --split                   # Split clean train/test
python cli.py --resize                  # Letterbox resize per tutte le architetture
python cli.py --pipeline                # Tutti e tre in sequenza

# Training
python cli.py --cross-validate --exp 6 --model resnet18
python cli.py --cross-validate --exp 6 --model convnext_tiny
python cli.py --cross-validate --exp 6 --model dinov2_base

# Fold singolo / aggregazione
python cli.py --cross-validate --exp 6 --model resnet50 --fold 0
python cli.py --cross-validate --exp 6 --model resnet50 --aggregate-only
```

---

## Stack Tecnologico

| Layer | Tecnologia |
|-------|-----------|
| Training | PyTorch 2.6 + CUDA 12.4, torchvision, Captum |
| Modelli | ResNet18/50, ConvNeXt-Tiny/Small, DINOv2-Small/Base + LoRA |
| Backend | FastAPI, Python 3.12 |
| Frontend | React 19, TypeScript, Vite, Tailwind v4, Shadcn/ui, Recharts |
| Infra | Podman + NVIDIA Container Toolkit, HuggingFace Spaces (Docker SDK) |

---

## Struttura Progetto

```
resnet18/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── models/           Registry multi-modello + classificatori
│   │   │   ├── kfold/            5-fold stratified cross-validation
│   │   │   ├── training.py       Training loop con gradient accumulation
│   │   │   └── evaluation.py     Metriche e valutazione
│   │   ├── preprocessing/        Cropper (Faster R-CNN), split, letterbox resize
│   │   ├── cli/                  Interfaccia CLI (experiments, runner, definitions)
│   │   ├── api/                  Endpoint FastAPI
│   │   ├── explainability/       Explainability con Captum
│   │   └── visualization/        Generazione grafici
│   ├── scripts/hf_tools/         Script upload/download HuggingFace
│   ├── experiments/              Risultati training e pesi modelli
│   ├── cli.py                    Entry point CLI
│   └── server.py                 Entry point server API
├── frontend/
│   ├── src/
│   │   ├── features/             Moduli explorer, classify, dataset
│   │   ├── pages/                Dashboard, Esperimenti, Classifica, Dataset, Ricerca
│   │   ├── locales/              Traduzioni EN + IT
│   │   └── components/           Layout + Shadcn/ui
│   └── package.json
├── Dockerfile
└── podman-start.sh
```

---

## API

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/classify/upload` | POST | Classifica immagine caricata (`model_id` opzionale) |
| `/classify/explain` | POST | Heatmap occlusione per immagine caricata |
| `/classify/models` | GET | Modelli addestrati disponibili con metriche |
| `/api/experiments` | GET | Lista tutti gli esperimenti |
| `/api/experiments/{id}` | GET | Dettaglio esperimento con dati fold |
| `/api/species` | GET | Lista specie con conteggio immagini |
| `/api/dataset/classes` | GET | Classi dataset con conteggi train/test |
| `/api/dataset/images/{split}/{class}/{file}` | GET | Serve immagine dataset |
| `/docs` | GET | Swagger UI |

---

## Link

- **GitHub:** [squidslab/OrchID](https://github.com/squidslab/OrchID) · [marcello-russo](https://github.com/marcello-russo)
- **Demo live:** [marcellorusso-orchid-ncd.hf.space](https://marcellorusso-orchid-ncd.hf.space)
- **Dataset:** `marcellorusso/orchid-ncd-dataset`

## Licenza

[MIT](LICENSE)
